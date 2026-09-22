"""P1-E: frozen-protocol, synthetic-only identifiability diagnostics.

No legacy experiment entry points are executed. Historical states are read-only.
"""
from __future__ import annotations
import argparse, copy, datetime, hashlib, inspect, json, math, os, subprocess, sys, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from acawlr_ppp_bridge import ReducedRankOneSyntheticPredictor as Model
from acawlr_ppp_synthetic import PENALTY, SCALE, quadrature
from acawlr_ppp_teacher_student import state_hash
ROOT=Path(__file__).resolve().parent
DT=torch.float64
TEACHER_HASH='b0d6db361a0a0c5e685d0e91f07601521760a7efd34e15c07f909ff4473338d8'
INIT_HASH='c94a02d1929dfa4d5ffbbfb55c8aaccf7d7d2d52f686203bf8e5c2eeffc064c6'
CUTS=(1e-6,1e-8,1e-10)

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def vec(m): return torch.cat([p.detach().flatten() for p in m.parameters()])
def setvec(m,v):
    i=0
    with torch.no_grad():
        for p in m.parameters(): p.copy_(v[i:i+p.numel()].reshape_as(p)); i+=p.numel()
def model(state):
    m=Model(0); m.load_state_dict(state,strict=True); return m.eval()
def penalty_parts(m):
    return dict(beta=float(.01*m.beta.detach().square().sum()),u=float(.1*m.u.detach().square().sum()),
                theta=float(.0005*sum(p.detach().square().sum() for n,p in m.named_parameters() if n not in ('beta','u'))))
def flat_grad(v,m):
    a=torch.autograd.grad(v,tuple(m.parameters()),allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for g,p in zip(a,m.parameters())])
@torch.no_grad()
def surface(m,xy,chunk=512):
    h=torch.cat([m.h(q) for q in xy.split(chunk)])
    b=m.beta+h[:,None]*m.u
    return dict(h=h,b=b,g=(xy/SCALE*b).sum(1))
def target(g,w):
    if g.ndim!=1 or w.shape!=g.shape or not torch.isfinite(g).all() or not torch.isfinite(w).all() or not (w>0).all():
        raise ValueError('Finite scores and strictly positive matching integration weights required')
    z=torch.logsumexp(w.log()+g,0)
    return dict(g=g.detach().clone(),w=w.detach().clone(),pi=(w.log()+g-z).exp().detach(),logz=z.detach())
def pop(g,t): return torch.logsumexp(t['w'].log()+g,0)-(t['pi']*g).sum()
def measures(g,t):
    z=torch.logsumexp(t['w'].log()+g,0); pi=(t['w'].log()+g-z).exp(); d=g-t['g']
    a=t['w']/t['w'].sum(); p=t['pi']
    return dict(logZ=float(z),KL=float((p*(t['g']-g)).sum()+z-t['logz']),TV=float((p-pi).abs().sum()/2),
                centered_g_area_RMSE=float((a*(d-(a*d).sum()).square()).sum().sqrt()),
                centered_g_teacher_RMSE=float((p*(d-(p*d).sum()).square()).sum().sqrt()))
def fidelity(a,b):
    errors=dict(logZ=abs(a['logZ']-b['logZ']),KL=abs(a['KL']-b['KL']),TV=abs(a['TV']-b['TV']))
    return dict(errors=errors,passed=errors['logZ']<=1e-4 and errors['KL']<=1e-6+.01*max(a['KL'],b['KL']) and errors['TV']<=1e-3)
def canonical(m,h,w):
    norm=float(m.u.detach().norm()); contribution=float(((h[:,None]*m.u.detach()).square().sum(1)*w/w.sum()).sum().sqrt())
    if contribution<=1e-12: return dict(defined=False,v=None,f=None,field_RMS=contribution)
    v=m.u.detach()/norm; sign=1 if float(v[torch.argmax(v.abs())])>=0 else -1
    return dict(defined=True,v=(sign*v).tolist(),f=sign*norm*h,field_RMS=contribution)
def global_functions(b):
    """log(sinh(b)/b), its first and second derivatives; stable at 0 and infinity."""
    b=np.asarray(b,dtype=float); a=np.abs(b); small=a<.05; s=np.where(small,b,0.); t=s*s
    z=t/6-t*t/180+t**3/2835-t**4/37800+t**5/467775
    mu=s/3-s**3/45+2*s**5/945-s**7/4725+2*s**9/93555
    var=1/3-t/15+2*t*t/189-t**3/675+2*t**4/10395
    safe=np.where(small,1.,a); e=np.exp(-2*safe)
    return (np.where(small,z,safe+np.log1p(-e)-np.log(2*safe)),
            np.where(small,mu,np.sign(b)*(1+2*e/(1-e)-1/safe)),
            np.where(small,var,1/safe**2-4*e/(1-e)**2))
def solve_global(moment,ridge):
    beta=np.array([brentq(lambda b:float(global_functions(b)[1])+ridge*b/256-float(t),-100,100,xtol=1e-14) for t in moment])
    residual=global_functions(beta)[1]+ridge*beta/256-np.asarray(moment)
    return dict(beta=beta.tolist(),moment_residual_inf=float(abs(residual).max()),converged=bool(abs(residual).max()<=1e-10))
def convex_newton(D,t,ridge):
    a=torch.zeros(D.shape[1],dtype=DT); lam=torch.tensor(ridge,dtype=DT)/256; hist=[]
    def value(v): return pop(D@v,t)+(lam*v.square()).sum()/2
    status='iteration_budget'
    for it in range(101):
        g=D@a; p=torch.softmax(t['w'].log()+g,0); mu=p@D
        grad=(p-t['pi'])@D+lam*a; centered=D-mu
        H=centered.T@(p[:,None]*centered)+torch.diag(lam)
        residual=float(grad.abs().max()); hist.append(dict(iteration=it,value=float(value(a)),residual_inf=residual))
        if residual<=1e-10: status='converged'; break
        if it==100: break
        e,V=torch.linalg.eigh(H); keep=e>e.max()*1e-10
        if not keep.any(): status='singular';break
        step=V[:,keep]@((V[:,keep].T@grad)/e[keep]); descent=grad@step
        for back in range(30):
            rate=.5**back; trial=a-rate*step
            if value(trial)<=value(a)-1e-4*rate*descent: a=trial;break
        else: status='line_search_failed';break
    return dict(parameters=a.tolist(),status=status,converged=status=='converged',gradient_inf=residual,history=hist,**measures(D@a,t))

def event(out,phase,status,scientific='not_evaluated',reason=None):
    memory=ROOT/'docs/project_memory'; active=read(memory/'ACTIVE_SESSION.json'); prov=read(out/'PROVENANCE.json')
    row=dict(event_id=f'{out.name}_{phase}_{status}_{utc()}',timestamp_utc=utc(),session_id=active['session_id'],run_id=out.name,
             phase=phase,status=status,command=prov['command'],output_dir=out.relative_to(ROOT).as_posix(),exit_code=None,
             source_base_commit=prov['HEAD'],source_dirty=prov['dirty'],source_manifest_path=(out/'PROVENANCE.json').relative_to(ROOT).as_posix(),
             source_manifest_sha256=sha(out/'PROVENANCE.json'),protocol_path=(out/'PROTOCOL.json').relative_to(ROOT).as_posix(),
             protocol_sha256=sha(out/'PROTOCOL.json'),provenance_path=(out/'PROVENANCE.json').relative_to(ROOT).as_posix(),
             artifacts=[p.relative_to(ROOT).as_posix() for p in out.glob('*.json')],scientific_status=scientific,blocker_or_failure_reason=reason)
    with (memory/'RUN_INDEX.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (memory/'sessions'/f"{active['session_id']}.md").open('a',encoding='utf-8') as f:
        f.write(f'\n- {utc()} {out.name}: {phase} {status}; scientific={scientific}; {reason or ""}\n')
    statusfile=out/'PHASE_STATUS.json'; stages=read(statusfile) if statusfile.exists() else {}
    stages[phase]=dict(status=status,scientific_status=scientific,reason=reason); write(statusfile,stages)
    text=(f'# P1-E 接续状态\n\n更新时间 UTC：{utc()}\n分支 codex-refactor；HEAD {prov["HEAD"]}。本轮源码和结果尚未提交；HEAD 不包含本轮实现。\n\n'
          '五层对象：密度 p、预测函数 g、系数场 b、beta/u/h 分解、神经参数 theta。当前主目标为 p/g；field identification 尚不能从有限网格自动推出。\n\n'
          f'P1-C/P1-D/P1-D LR frozen；本轮阶段状态：\n```json\n{json.dumps(stages,ensure_ascii=False,indent=2)}\n```\n\n'
          f'最近运行：{out.name}；命令 `{prov["command"]}`；进程退出状态见最终 session，进行中不可视为成功。\n\n'
          f'证据入口：{out.relative_to(ROOT).as_posix()}/PROTOCOL.json、PROVENANCE.json、STRUCTURE.json、CONVEX_CONTROLS.json、JACOBIAN.json；尚未存在者为 not_created。\n\n'
          '最小下一步：只执行冻结协议允许且通过 gate 的阶段；失败不放宽阈值、不重训、不进入真实数据。旧解释未排除项：conditioning、盆地、正则偏差、有限样本、积分误差；不能仅凭 branch 非零判断成功。\n\n'
          f'阅读顺序：sessions/{active["session_id"]}.md → 本轮 protocol/provenance → SUMMARY/NOTES（如已生成）→ raw → 源码。\n')
    tmp=memory/'CATCH_UP.tmp'; tmp.write_text(text,encoding='utf-8'); os.replace(tmp,memory/'CATCH_UP.md')

def reserve_output(out):
    out.mkdir(parents=True,exist_ok=False)
    for name in ('raw','logs','figures'): (out/name).mkdir()

def protocol():
    prompt=next((ROOT/'docs/project_memory/prompts').glob('*_P1E_IDENTIFIABILITY_CODEX_PROMPT_WITH_PROJECT_MEMORY.md'))
    return dict(version='P1-E-v1',frozen_at=utc(),task_spec=prompt.relative_to(ROOT).as_posix(),task_sha256=sha(prompt),
      complete_spec=prompt.read_text(encoding='utf-8'),grids=[24,48,96,192],n_ref=256,dtype='float64',threads=1,
      independent_points=dict(seed=20260917,count=257,extra='origin and +/-1e-6 coordinate-axis points'),
      line_scan=dict(directions=181,angles='j*pi/181, j=0..180',points=65,special_line_points=129),
      FD_eps=[1e-4,1e-5,1e-6],collapsed_directions=dict(seed=20260919,count=3,steps=[1e-2,1e-3,1e-4]),
      svd_cutoffs=CUTS,whitening_cutoff=1e-10,weak_directions='smallest three generalized eigenvalues in positive K subspace; tie order from eigh; largest-magnitude coordinate positive',
      whitening_convention='relative singular-value cutoff of area-weighted Jb; corresponding K eigenvalue cutoff squared',weak_steps=[1e-4,1e-3,1e-2],weak_norm='step * max(1, parameter norm)',
      convex=dict(iterations=100,armijo=1e-4,backtracks=30,svd_cutoff=1e-10,tolerance=1e-10),
      population=dict(cold_starts=2,oracle_max=1,oracle_seed=20260918,steps=2000,lr=.003,betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,
      checkpoints=[0,100,300,600,1000,1500,2000],observe_every=25,penalty=vars(PENALTY),training_grid=48),
      tolerances=dict(copy=1e-12,KL_identity_atol=1e-10,KL_identity_rtol=1e-10,teacher_gradient=1e-9,fd_atol=1e-7,fd_rtol=1e-4,
      logZ_fidelity=1e-4,KL_fidelity_atol=1e-6,KL_fidelity_rtol=.01,TV_fidelity=1e-3,density_KL=1e-4,density_TV=.01,predictor_RMSE=.02,
      equivalence_KL=1e-10,equivalence_g=1e-6,field_zero=1e-12,stationary_gradient_per_n=1e-3,stationary_range_per_n=1e-5),
      gate_policy='Conservative: any required A/B/C implementation or 96-to-192 fidelity failure (including archived fixed endpoints) blocks all new neural trajectories. Static diagnostics still finish. No threshold changes or extra training.',
      oracle_floor=1e-10,oracle_reduction=.9)

def setup(out):
    reserve_output(out); write(out/'PROTOCOL.json',protocol())
    active=read(ROOT/'docs/project_memory/ACTIVE_SESSION.json'); frozen=read(ROOT/active['frozen_manifest'])
    assert all(sha(ROOT/p)==h for p,h in frozen.items()),'Frozen file changed before run'
    sources={p.name:sha(p) for p in ROOT.glob('*p1e*.py')}
    prov=dict(start=utc(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
              dirty=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),sources=sources,frozen=frozen,
              torch=str(torch.__version__),python=sys.version,executable=sys.executable,threads=1,dtype='float64',
              command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),protocol_sha256=sha(out/'PROTOCOL.json'))
    write(out/'PROVENANCE.json',prov); write(out/'POPULATION_RESULTS.json',dict(status='not_started',trajectories=[],count=0))
    return prov

def load_archives(out,prov):
    archive=torch.load(ROOT/'ACAWLR_PPP_P1C_TS_STATES.pt',weights_only=True,map_location='cpu')
    lr=torch.load(ROOT/'ACAWLR_PPP_P1D_LR_STATES.pt',weights_only=True,map_location='cpu')
    teacher=model(archive['teacher_state']).requires_grad_(False)
    initial=archive['cases']['B_n256_data23_init1011']['initial_state']
    assert state_hash(teacher.state_dict())==TEACHER_HASH
    assert state_hash(initial)==INIT_HASH
    assert state_hash(lr['teacher_state'])==TEACHER_HASH
    prov.update(teacher_sha256=TEACHER_HASH,init_sha256=INIT_HASH,
        events={k:state_hash({'events':v['events']}) for k,v in lr['cases'].items()},
        parameter_order=[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in teacher.named_parameters()])
    assert sum(p.numel() for p in teacher.parameters())==675
    points=(2*torch.rand((257,2),generator=torch.Generator().manual_seed(20260917),dtype=DT)-1)*SCALE
    points=torch.cat([points,torch.zeros(1,2,dtype=DT),torch.eye(2,dtype=DT)*SCALE*1e-6,-torch.eye(2,dtype=DT)*SCALE*1e-6])
    torch.save(points,out/'raw/independent_points.pt')
    grids={}; targets={}; surfaces={}
    for r in (24,48,96,192):
        xy,w=quadrature(r); s=surface(teacher,xy); t=target(s['g'],w); grids[r]=(xy,w); targets[r]=t; surfaces[r]=s
        torch.save(dict(xy=xy,weights=w,teacher=s,target=t),out/f'raw/grid_{r}.pt')
    prov['grid_hashes']={str(r):state_hash(dict(xy=q,weights=w)) for r,(q,w) in grids.items()}
    write(out/'PROVENANCE.json',prov)
    return teacher,initial,lr,points,grids,targets,surfaces

@torch.no_grad()
def compare_surface(a,b,xy,w):
    x,y=surface(a,xy),surface(b,xy); t=target(x['g'],w); d=y['g']-x['g']; p=t['pi']
    return dict(g_max=float(d.abs().max()),aligned_g_max=float((d-(p*d).sum()).abs().max()),
                field_max=float((y['b']-x['b']).abs().max()),p_max=float(((y['g']-torch.logsumexp(w.log()+y['g'],0)).exp()-(x['g']-t['logz']).exp()).abs().max()),
                penalty_delta=float(PENALTY(b)-PENALTY(a)),**measures(y['g'],t))

def symmetries(teacher,points,grids):
    out={}
    for name in ('copy','sign','relu_scale','permutation','axial_sign','naive_readout_scale'):
        m=model(teacher.state_dict())
        with torch.no_grad():
            if name=='sign': m.u.neg_();m.cawnn.readout.weight.neg_()
            if name=='relu_scale':
                m.aspnn.network[0].weight[0]*=2; m.aspnn.network[0].bias[0]*=2; m.aspnn.network[2].weight[:,0]/=2
            if name=='permutation':
                idx=torch.arange(16);idx[0]=1;idx[1]=0
                m.aspnn.network[0].weight.copy_(m.aspnn.network[0].weight[idx]);m.aspnn.network[0].bias.copy_(m.aspnn.network[0].bias[idx]);m.aspnn.network[2].weight.copy_(m.aspnn.network[2].weight[:,idx])
            if name=='axial_sign': m.aspnn.network[0].weight.neg_()
            if name=='naive_readout_scale': m.u.mul_(2); m.cawnn.readout.weight.div_(2)
        out[name]={str(r):compare_surface(teacher,m,*grids[r]) for r in (48,96)}
        out[name]['independent']=compare_surface(teacher,m,points,torch.ones(len(points),dtype=DT))
        out[name]['exact_symmetry_expected']=name!='naive_readout_scale'
    return out

@torch.no_grad()
def line_diagnostics(teacher,out):
    a=torch.tensor([-1.2*math.sin(.35),.8*math.cos(.35)],dtype=DT)
    def line(normal,n):
        d=torch.stack([-normal[1],normal[0]]);d=d/d.abs().max()
        x=torch.linspace(-1,1,n,dtype=DT)[:,None]*d
        h=surface(teacher,x*SCALE)['h']
        return dict(x=x,h=h,radius=x.norm(dim=1))
    raw=dict(a_zero=line(a,129),u_zero=line(teacher.u,129)); rows=[]
    for j in range(181):
        angle=j*math.pi/181; d=torch.tensor([math.cos(angle),math.sin(angle)],dtype=DT)
        z=line(torch.stack([-d[1],d[0]]),65); rows.append(dict(angle=angle,**z))
    raw['scan']=rows;torch.save(raw,out/'raw/lines.pt')
    def stats(z):return dict(RMS=float(z['h'].square().mean().sqrt()),max_abs=float(z['h'].abs().max()),max_radius=float(z['radius'].max()))
    return dict(a=a.tolist(),a_zero=stats(raw['a_zero']),u_zero=stats(raw['u_zero']),scan=[dict(angle=z['angle'],**stats(z)) for z in rows],
        limitation='Finite point scans neither prove a continuous zero line nor exclude unsampled zero lines.')

def stage_a(teacher,initial,lr,points,grids,targets,surfaces,out):
    result=dict(symmetries=symmetries(teacher,points,grids),lines=line_diagnostics(teacher,out),penalty=penalty_parts(teacher),historical={})
    m=model(teacher.state_dict());xy,w=grids[48];t=targets[48]
    likelihood=pop(m.g(xy),t); grad=flat_grad(likelihood,m)
    pg=flat_grad(PENALTY(m),m)
    result['teacher_identity']=dict(gap=float(likelihood.detach()-pop(t['g'],t)),gradient_max=float(grad.abs().max()),penalty_only_gradient_max=float(pg.abs().max()))
    shrink=[]; base=256*pop(t['g'],t)+PENALTY(teacher)
    q=surfaces[48]['g']-(xy/SCALE)@teacher.beta
    for eps in (1e-4,1e-5,1e-6):
        with torch.no_grad(): m.u.copy_((1-eps)*teacher.u)
        plus=256*pop(m.g(xy),t)+PENALTY(m)
        with torch.no_grad(): m.u.copy_((1+eps)*teacher.u)
        minus=256*pop(m.g(xy),t)+PENALTY(m)
        derivative=float((plus-minus)/(2*eps)); shrink.append(dict(epsilon=eps,central=derivative,forward=float((plus-base)/eps),analytic=-9.,passed=abs(derivative+9)<=1e-7+1e-4*9))
    result['shrink']=dict(checks=shrink,likelihood_second_derivative=float(256*(t['pi']*(q-(t['pi']*q).sum()).square()).sum()))
    # Independent smooth beta direction tests the true differentiable population forward.
    m=model(initial);direction=torch.tensor([.6,-.8],dtype=DT);g=m.g(xy);derivative=float((torch.autograd.grad(pop(g,t),m.beta)[0]*direction).sum()); origin=m.beta.detach().clone();fd=[]
    for eps in (1e-4,1e-5,1e-6):
        with torch.no_grad():m.beta.copy_(origin+eps*direction)
        plus=pop(m.g(xy),t).detach()
        with torch.no_grad():m.beta.copy_(origin-eps*direction)
        minus=pop(m.g(xy),t).detach()
        d=float((plus-minus)/(2*eps));fd.append(dict(epsilon=eps,analytic=derivative,numeric=d,passed=abs(d-derivative)<=1e-7+1e-4*abs(derivative)))
    result['smooth_FD']=fd
    at_origin=model(teacher.state_dict());x=torch.zeros((1,2),dtype=DT,requires_grad=True);go=at_origin.g(x*SCALE)
    result['anchor']=dict(h0=float(at_origin.h(at_origin.reference).detach()),g0=float(go.detach()),gradient_x=torch.autograd.grad(go.sum(),x)[0].tolist(),beta=teacher.beta.tolist())
    for key,case in lr['cases'].items():
        m=model(case['checkpoints'][2000]['model_state']);rows={}
        for r in (48,96,192):
            s=surface(m,grids[r][0]);rows[str(r)]=measures(s['g'],targets[r]);torch.save(s,out/f'raw/legacy_{key}_{r}.pt')
        rows['fidelity']=fidelity(rows['96'],rows['192']);rows['penalty']=penalty_parts(m)
        old48=case['checkpoints'][2000]['g_surface']
        rows['archive_forward_max_abs']=float((surface(m,grids[48][0])['g']-old48).abs().max())
        if key=='0.001_71':
            z24=measures(surface(m,grids[24][0])['g'],targets[24])['logZ'];d=rows['48']['logZ']-z24
            rows['24_to_48']=dict(logZ24=z24,delta_logZ=d,SUM_correction=256*d)
        result['historical'][key]=rows
        print('historical',key,rows['fidelity'],flush=True)
    result['teacher_fidelity']=fidelity(measures(targets[96]['g'],targets[96]),measures(targets[192]['g'],targets[192]))
    result['KL_identity']={}
    for key in ('0.01_23','0.01_71','0.001_71'):
        m=model(lr['cases'][key]['checkpoints'][2000]['model_state']);g=surface(m,xy)['g'];kl=measures(g,t)['KL'];gap=float(pop(g,t)-pop(t['g'],t))
        result['KL_identity'][key]=dict(gap=gap,KL=kl,passed=math.isclose(gap,kl,rel_tol=1e-10,abs_tol=1e-10))
    write(out/'STRUCTURE.json',result);return result

def stage_b(teacher,grids,targets,surfaces,out):
    result={}; raw={}
    for r in (48,96,192):
        xy,w=grids[r];t=targets[r];X=xy/SCALE; h=surfaces[r]['h'];D=torch.cat([X,h[:,None]*X],1);mom=t['pi']@X
        z=float(t['logz']);Eg=float(t['pi']@t['g']);rows=dict(teacher=dict(moment=mom.tolist(),Eg=Eg,logZ=z,penalty=penalty_parts(teacher)))
        for name,ridge in (('global',0.),('global_ridge',.02)):
            row=solve_global(mom.numpy(),ridge);b=torch.tensor(row['beta'],dtype=DT);zg=math.log(9600)+float(global_functions(row['beta'])[0].sum())
            kl=Eg-float(mom@b)+zg-z; p=(X@b-zg).exp()*w
            row.update(logZ=zg,KL=kl,TV=float((p-t['pi']).abs().sum()/2),information44=44*kl,information256=256*kl,
                       penalty=dict(beta=.01*float(b.square().sum()) if ridge else .01*float(b.square().sum()),u=0.,theta=0.))
            row['penalized_minus_teacher']=256*kl+sum(row['penalty'].values())-float(PENALTY(teacher)); rows[name]=row
        for name,ridge in (('fixed_h',[0.,0.,0.,0.]),('fixed_h_ridge',[.02,.02,.2,.2])):
            row=convex_newton(D,t,ridge);a=torch.tensor(row['parameters'],dtype=DT)
            row['penalty']=dict(beta=.01*float(a[:2].square().sum()),u=.1*float(a[2:].square().sum()),theta=penalty_parts(teacher)['theta'])
            row['penalized_minus_teacher']=256*row['KL']+sum(row['penalty'].values())-float(PENALTY(teacher));rows[name]=row
        centered=D-t['pi']@D;s=torch.linalg.svdvals(t['pi'].sqrt()[:,None]*centered)
        rows['fixed_h_design']=dict(singular_values=s.tolist(),ranks={str(c):int((s>s[0]*c).sum()) for c in CUTS},condition=float(s[0]/s[-1]))
        q=h*(X@teacher.u);proj={};A=torch.cat([torch.ones(len(X),1,dtype=DT),X],1)
        for name,weight in (('area',w/w.sum()),('teacher',t['pi'])):
            c=torch.linalg.lstsq(weight.sqrt()[:,None]*A,weight.sqrt()*q).solution;res=q-A@c
            rms=(weight*res.square()).sum().sqrt();stdq=(weight*(q-weight@q).square()).sum().sqrt();stdg=(weight*(t['g']-weight@t['g']).square()).sum().sqrt()
            proj[name]=dict(coefficients=c.tolist(),residual_RMS=float(rms),ratio_std_q=float(rms/stdq),ratio_std_g=float(rms/stdg))
        rows['projection']=proj;result[str(r)]=rows;raw[r]=dict(design=D,singular_values=s)
        print('controls',r,'global gap',rows['global']['KL'],'fixed-h',rows['fixed_h_ridge']['status'],flush=True)
    result['fidelity']={name:fidelity(result['96'][name],result['192'][name]) for name in ('global','global_ridge','fixed_h','fixed_h_ridge')}
    torch.save(raw,out/'raw/convex.pt');write(out/'CONVEX_CONTROLS.json',result);return result

def collapsed_check(teacher,grids,targets,controls,out):
    xy,w=grids[48];t=targets[48];result={};raw={}
    # Discrete-global Newton supplies beta for the *same quadrature objective*.
    for name,lam in (('unpenalized',0.),('penalized',1.)):
        m=model(teacher.state_dict())
        for p in m.parameters():p.data.zero_()
        opt=convex_newton(xy/SCALE,t,[.02*lam]*2)
        with torch.no_grad():m.beta.copy_(torch.tensor(opt['parameters'],dtype=DT))
        base=vec(m);f=lambda:256*pop(m.g(xy),t)+lam*PENALTY(m)
        value=f();gradient=flat_grad(value,m);rows=[];rng=torch.Generator().manual_seed(20260919);directions=[]
        for j in range(3):
            d=torch.randn(base.shape,generator=rng,dtype=DT);d=d/d.norm();directions.append(d)
            for step in (1e-2,1e-3,1e-4):
                for sign in (-1,1):
                    setvec(m,base+sign*step*max(1,float(base.norm()))*d)
                    rows.append(dict(direction=j,step=sign*step,objective_delta=float(f().detach()-value.detach())))
        setvec(m,base);raw[name]=dict(state=m.state_dict(),directions=torch.stack(directions))
        result[name]=dict(beta=opt['parameters'],global_solver=opt['status'],gradient_max=float(gradient.abs().max()),h_max=float(surface(m,xy)['h'].abs().max()),displacements=rows)
    torch.save(raw,out/'raw/collapsed.pt');write(out/'COLLAPSED.json',result);return result

class HWrapper(torch.nn.Module):
    def __init__(self,m):super().__init__();self.core=m
    def forward(self,xy):return self.core.h(xy)

def jacobians(m,xy):
    wrapper=HWrapper(m); params=dict(wrapper.named_parameters()); js=[]
    for chunk in xy.split(32):
        jac=torch.func.jacrev(lambda p:torch.func.functional_call(wrapper,p,(chunk,)),chunk_size=8)(params)
        js.append(torch.cat([jac[n].reshape(len(chunk),-1) for n in params],1).detach())
    Jh=torch.cat(js);s=surface(m,xy);X=xy/SCALE;N=len(xy);P=Jh.shape[1]
    Jg=(X@m.u.detach())[:,None]*Jh;Jg[:,:2]+=X;Jg[:,2:4]+=s['h'][:,None]*X
    Jb=m.u.detach()[None,:,None]*Jh[:,None,:]
    Jb[:,:,:2]+=torch.eye(2,dtype=DT)[None,:,:]
    Jb[:,:,2:4]+=s['h'][:,None,None]*torch.eye(2,dtype=DT)[None,:,:]
    return Jh,Jg,Jb.reshape(N*2,P)

def stage_c(teacher,initial,lr,grids,targets,points,out):
    result={};xy,w=grids[48];t=targets[48];area=w/w.sum()
    cases=dict(teacher=teacher.state_dict(),init1011=initial,collapsed=lr['cases']['0.01_23']['checkpoints'][2000]['model_state'])
    for name,state in cases.items():
        print('Jacobian start',name,flush=True);m=model(state);Jh,Jg,Jb=jacobians(m,xy);centered=Jg-t['pi']@Jg
        A=t['pi'].sqrt()[:,None]*centered; B=area.repeat_interleave(2).sqrt()[:,None]*Jb
        F=A.T@A;K=B.T@B
        sg=torch.linalg.svdvals(A);_,sb,Vh=torch.linalg.svd(B,full_matrices=False)
        spectra={};basis=None;directions=None
        for cut in CUTS:
            keep=sb>sb[0]*cut;W=Vh[keep].T/sb[keep]
            small=(A@W).T@(A@W);ev,U=torch.linalg.eigh(small)
            spectra[str(cut)]=dict(eigenvalues=ev.tolist(),K_rank=int(keep.sum()))
            if cut==1e-10:
                basis=W;directions=(W@U[:,:min(3,len(ev))]).T
        base=vec(m);directions=directions/directions.norm(dim=1)[:,None]
        for d in directions:
            if d[d.abs().argmax()]<0:d.neg_()
        raw=dict(Jh=Jh,Jg=Jg,Jb=Jb,F=F,K=K,g_singular=sg,b_singular=sb,generalized=spectra,directions=directions,parameter_vector=base)
        torch.save(raw,out/f'raw/jacobian_{name}.pt')
        # Check chain-rule Jacobian against independent full-forward AD on fixed points.
        wrapper=HWrapper(m)
        class GWrapper(torch.nn.Module):
            def __init__(self,core):super().__init__();self.core=core
            def forward(self,p):return self.core.g(p)
        gw=GWrapper(m);par=dict(gw.named_parameters());jj=torch.func.jacrev(lambda p:torch.func.functional_call(gw,p,(xy[:3],)))(par)
        direct=torch.cat([jj[n].reshape(3,-1) for n in par],1)
        chain_error=float((direct-Jg[:3]).abs().max())
        rows=[];bases={str(r):surface(m,grids[r][0]) for r in (48,96)};bases['independent']=surface(m,points)
        for j,d in enumerate(directions):
            for step in (1e-4,1e-3,1e-2):
                for sign in (-1,1):
                    setvec(m,base+sign*step*max(1,float(base.norm()))*d);row=dict(direction=j,relative_step=sign*step,penalty=penalty_parts(m),evaluations={})
                    for key,coords,weight in [(str(r),*grids[r]) for r in (48,96)]+[('independent',points,torch.ones(len(points),dtype=DT))]:
                        s=surface(m,coords);old=bases[key];tt=target(old['g'],weight);ds=s['g']-old['g'];weights=weight/weight.sum()
                        row['evaluations'][key]=dict(**measures(s['g'],tt),raw_g_RMS=float((weights*ds.square()).sum().sqrt()),
                          field_RMS=float((weights[:,None]*(s['b']-old['b']).square()).sum().sqrt()),
                          aligned_g_max=float((ds-(tt['pi']*ds).sum()).abs().max()))
                    rows.append(row)
        setvec(m,base)
        # Directional linear sensitivities are measured separately from finite alternatives.
        sens=[dict(field_norm=float((B@d).norm()),density_norm=float((A@d).norm())) for d in directions]
        result[name]=dict(status='COMPLETED',chain_rule_max_error=chain_error,
            ranks={str(c):dict(g=int((sg>sg[0]*c).sum()),field=int((sb>sb[0]*c).sum())) for c in CUTS},
            generalized=spectra,directional_sensitivity=sens,displacements=rows,
            limitation='Taylor-null is not an exact finite alternative; no global-identification conclusion.')
        write(out/'JACOBIAN.json',result);event(out,'C_'+name,'COMPLETED','passed' if chain_error<=1e-9 else 'not_met')
        print('Jacobian complete',name,result[name]['ranks'],flush=True)
    result['grid24_dimension_bound']=dict(nodes=576,probability_rank_max=575,parameters=675,nullity_min=100)
    write(out/'JACOBIAN.json',result);return result

def gates(a,b,c,collapsed):
    g={}
    g['teacher_copy']=max(a['symmetries']['copy']['96'][k] for k in ('g_max','field_max','p_max'))<=1e-12
    g['teacher_gradient']=a['teacher_identity']['gradient_max']<=1e-9
    g['teacher_identity']=abs(a['teacher_identity']['gap'])<=1e-10
    g['symmetries']=all(a['symmetries'][n]['96']['aligned_g_max']<=1e-6 and a['symmetries'][n]['96']['KL']<=1e-10 for n in ('sign','relu_scale','permutation','axial_sign'))
    g['FD']=all(x['passed'] for x in a['smooth_FD']) and all(x['passed'] for x in a['shrink']['checks'])
    g['KL_identity']=all(x['passed'] for x in a['KL_identity'].values())
    g['teacher_fidelity']=a['teacher_fidelity']['passed']
    g['archived_forward']=all(x['archive_forward_max_abs']<=1e-12 for x in a['historical'].values())
    for k,v in a['historical'].items():g['fidelity_legacy_'+k]=v['fidelity']['passed']
    for k,v in b['fidelity'].items():g['fidelity_'+k]=v['passed']
    g['convex_solvers']=all(b[str(r)][name]['converged'] for r in (48,96,192) for name in ('global','global_ridge','fixed_h','fixed_h_ridge'))
    g['jacobian']=all(c[n]['chain_rule_max_error']<=1e-9 for n in ('teacher','init1011','collapsed'))
    g['collapsed_stationary']=all(v['gradient_max']/256<=1e-9 for v in collapsed.values())
    return dict(checks=g,passed=all(g.values()),failed=[k for k,v in g.items() if not v])

def branch_diagnostics(m,xy,t,teacher):
    with torch.no_grad():
        s=surface(m,xy);truth=surface(teacher,xy);a=t['w']/t['w'].sum();raw=m.raw_spatial(xy);ref=m.raw_spatial(m.reference);q=s['h']*(xy/SCALE@m.u)
        def stats(z):return dict(min=float(z.min()),max=float(z.max()),RMS=float((a*z.square()).sum().sqrt()),std=float((a*(z-a@z).square()).sum().sqrt()))
        canon=canonical(m,s['h'],t['w']);tc=canonical(teacher,truth['h'],t['w'])
        field_error=s['b']-truth['b'];fr=float((a[:,None]*field_error.square()).sum().sqrt());den=float((a[:,None]*truth['b'].square()).sum().sqrt())
        recovery=dict(field_RMSE=fr/math.sqrt(2),field_relative_RMSE=fr/den,components=[float((a*field_error[:,i].square()).sum().sqrt()) for i in (0,1)],
                      theta_distance=float((vec(m)[4:]-vec(teacher)[4:]).norm()),canonical_defined=canon['defined'])
        if canon['defined']:
            v=torch.tensor(canon['v'],dtype=DT);tv=torch.tensor(tc['v'],dtype=DT);recovery.update(v=canon['v'],direction_angle=float(torch.acos((v@tv).clamp(-1,1))),f_RMSE=float((a*(canon['f']-tc['f']).square()).sum().sqrt()))
        result=dict(raw_range=[float(raw.min()),float(raw.max())],tanh_range=[float(raw.tanh().min()),float(raw.tanh().max())],
            saturation_fraction=float((raw.tanh().abs()>=.99).double().mean()),saturation_definition='abs(tanh(raw))>=.99',reference_raw=float(ref[0]),u_norm=float(m.u.norm()),h=stats(s['h']),q=stats(q),recovery=recovery)
        # Explicit forward to expose all hidden ReLU activations, including functional residual ReLU.
        delta=m.anchors[None]-xy[:,None,None,:];angle=m.angle
        offsets=torch.stack(((delta[...,0]*angle.cos()+delta[...,1]*angle.sin())/150000,(-delta[...,0]*angle.sin()+delta[...,1]*angle.cos())/50000),-1).clamp(-20,20).reshape(-1,2)
        activations={};proximity=[]
        for sign in (-1,1):
            z=sign*offsets
            for i,layer in enumerate(m.aspnn.network):
                z=layer(z)
                if isinstance(layer,torch.nn.ReLU):activations[f'aspnn_{sign}_{i}']=float((z>0).double().mean())
            proximity.append(z)
        z=m.cawnn.head(((proximity[0]+proximity[1])/2).reshape(len(xy),1,4,3));rz=m.cawnn.residual[0](z)
        activations['residual_hidden']=float((rz>0).double().mean());z=torch.relu(z+m.cawnn.residual[2](rz.relu()));activations['residual_output']=float((z>0).double().mean())
        for name,pool in [('mean',z.mean((2,3),keepdim=True)),('max',z.amax((2,3),keepdim=True))]:activations['channel_'+name]=float((m.cawnn.channel[0](pool)>0).double().mean())
        result['ReLU_active_fractions']=activations
    # Location/reference contributions to the exact data gradient; autograd.grad does not alter .grad.
    raw=m.raw_spatial(xy);ref=m.raw_spatial(m.reference);g=(xy/SCALE@m.beta)+(xy/SCALE@m.u)*(raw.tanh()-ref.tanh())
    residual=(torch.softmax(t['w'].log()+g.detach(),0)-t['pi'])*256
    location=flat_grad((residual*(xy/SCALE@m.u.detach())*raw.tanh()).sum(),m)
    reference=flat_grad(-(residual*(xy/SCALE@m.u.detach())).sum()*ref.tanh().sum(),m)
    result['branch_data_gradient']=dict(location_theta_L2=float(location[4:].norm()),reference_theta_L2=float(reference[4:].norm()),sum_theta_L2=float((location[4:]+reference[4:]).norm()))
    return result

def population_run(label,state,lam,teacher,grids,targets,out):
    m=model(state);opt=torch.optim.Adam(m.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    xy,w=grids[48];t=targets[48];history=[];observations=[];previous=vec(m);checkpoints={};status='COMPLETED'
    for step in range(2001):
        opt.zero_grad(set_to_none=True);g=m.g(xy);data=256*pop(g,t);loss=data+lam*PENALTY(m);loss.backward()
        current=vec(m);gr=torch.cat([p.grad.flatten() for p in m.parameters()]);update=current-previous
        row=dict(step=step,data=float(data.detach()),total=float(loss.detach()),penalty=penalty_parts(m),
                 gradient_L2=float(gr.norm()),block_gradient=[float(gr[:2].norm()),float(gr[2:4].norm()),float(gr[4:].norm())],
                 block_update=[float(update[:2].norm()),float(update[2:4].norm()),float(update[4:].norm())])
        if not bool(torch.isfinite(loss)) or not torch.isfinite(gr).all():
            torch.save(dict(step=step,model=m.state_dict(),optimizer=opt.state_dict()),out/f'raw/{label}_nonfinite.pt');status='FAILED';break
        history.append(row)
        if step%25==0:
            observations.append(dict(step=step,branch=branch_diagnostics(m,xy,t,teacher),**{str(r):measures(surface(m,grids[r][0])['g'],targets[r]) for r in (48,96)}))
        if step in (0,100,300,600,1000,1500,2000):
            checkpoints[step]=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
            torch.save(checkpoints,out/f'raw/{label}_states.pt');write(out/f'raw/{label}_history.json',history);write(out/f'raw/{label}_observations.json',observations)
            print(label,step,row['total'],flush=True)
        if step==2000:break
        previous=current.clone();opt.step()
    final={str(r):measures(surface(m,grids[r][0])['g'],targets[r]) for r in (96,192)};check=fidelity(final['96'],final['192'])
    for r in (48,96,192):torch.save(surface(m,grids[r][0]),out/f'raw/{label}_{r}.pt')
    return dict(penalty=penalty_parts(m),label=label,status=status,steps=step,initial_KL=observations[0]['96']['KL'],final=final,fidelity=check,
        density_success=status=='COMPLETED' and check['passed'] and final['192']['KL']<=1e-4 and final['192']['TV']<=.01,
        predictor_success=final['192']['centered_g_area_RMSE']<=.02 and final['192']['centered_g_teacher_RMSE']<=.02,
        stationary=max(x['gradient_L2'] for x in history[-200:])/256<=1e-3 and (max(x['total'] for x in history[-200:])-min(x['total'] for x in history[-200:]))/256<=1e-5)

def run_population(initial,teacher,grids,targets,out,gate,requested):
    report=dict(status='BLOCKED' if not gate['passed'] else 'SKIPPED',reason=gate['failed'] if not gate['passed'] else 'CLI did not request population',count=0,trajectories=[])
    if gate['passed'] and requested:
        for label,lam in (('P0',0.),('P_lambda',1.)):
            report['trajectories'].append(population_run(label,initial,lam,teacher,grids,targets,out));report['count']+=1
            write(out/'POPULATION_RESULTS.json',report);event(out,label,report['trajectories'][-1]['status'])
        p0=report['trajectories'][0]
        if p0['status']=='COMPLETED' and not p0['density_success'] and p0['fidelity']['passed']:
            m=model(teacher.state_dict());v=vec(m);d=torch.randn(v.shape,generator=torch.Generator().manual_seed(20260918),dtype=DT);setvec(m,v+1e-3*max(1,float(v.norm()))*d/d.norm())
            o=population_run('oracle_local',m.state_dict(),0.,teacher,grids,targets,out)
            o['interpretation']='stability_only' if o['initial_KL']<1e-9 else 'local_recovery_probe';o['excess_reduction90']=o['final']['96']['KL']<=.1*o['initial_KL']
            report['trajectories'].append(o);report['count']+=1
        report['status']='COMPLETED';report['reason']=None
    write(out/'POPULATION_RESULTS.json',report);event(out,'D',report['status'],'not_evaluated' if report['count']==0 else 'not_met',str(report['reason']))
    return report

def run(out,requested):
    torch.set_num_threads(1);prov=setup(out);event(out,'audit','IN_PROGRESS')
    try:
        teacher,initial,lr,points,grids,targets,surfaces=load_archives(out,prov)
        event(out,'A','IN_PROGRESS');a=stage_a(teacher,initial,lr,points,grids,targets,surfaces,out);event(out,'A','COMPLETED','numerical_unresolved' if any(not x['fidelity']['passed'] for x in a['historical'].values()) else 'passed')
        event(out,'B','IN_PROGRESS');b=stage_b(teacher,grids,targets,surfaces,out);cl=collapsed_check(teacher,grids,targets,b,out);event(out,'B','COMPLETED','passed' if all(x['passed'] for x in b['fidelity'].values()) else 'numerical_unresolved')
        event(out,'C','IN_PROGRESS');c=stage_c(teacher,initial,lr,grids,targets,points,out);event(out,'C','COMPLETED','passed')
        gate=gates(a,b,c,cl);write(out/'GATES.json',gate);run_population(initial,teacher,grids,targets,out,gate,requested)
        event(out,'run','COMPLETED','passed' if gate['passed'] else 'numerical_unresolved')
    except BaseException:
        (out/'logs/error.txt').write_text(traceback.format_exc(),encoding='utf-8');event(out,'run','FAILED','not_evaluated','See logs/error.txt');raise
    finally:
        integrity={p:sha(ROOT/p)==h for p,h in prov['frozen'].items()};write(out/'FROZEN_VERIFY.json',dict(passed=all(integrity.values()),files=integrity,time=utc()))
        write(out/'EXECUTION_END.json',dict(ended=utc()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--run-population',action='store_true');args=p.parse_args()
    run(args.output_dir.resolve(),args.run_population)
