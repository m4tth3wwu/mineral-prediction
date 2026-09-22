"""P1-F population causal discrimination; independent frozen protocol and artifacts.

Reuses only pure P1-E numerical helpers and the unchanged bridge model.
Never calls P1-E run/gates/population_run or legacy training entrypoints.
"""
from __future__ import annotations
import argparse,copy,csv,datetime,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import acawlr_ppp_p1e_identifiability as e
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
MEM=REPO/'docs/project_memory'
OLD=ROOT/'p1e_identifiability/20260917T112555Z_p1e_v1'
CHECKPOINTS=(0,100,300,600,1000,1500,2000)
DT=torch.float64

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def atomic(p,v):
    tmp=Path(str(p)+'.tmp');write(tmp,v);os.replace(tmp,p)
def rel(p):return Path(p).relative_to(REPO).as_posix()
def tensor_load(p):return torch.load(p,weights_only=True,map_location='cpu')
def frozen_check():
    s=read(MEM/'P1F_SESSION.json');files=read(REPO/s['frozen_manifest']);bad=[p for p,h in files.items() if sha(REPO/p)!=h]
    return dict(time=utc(),count=len(files),passed=not bad,changed=bad)
def memory(out,phase,status,scientific='not_evaluated',reason=None,command=None,exit_code=None):
    session=read(MEM/'P1F_SESSION.json');prov=read(out/'PROVENANCE.json');p=out/'PROTOCOL.json'
    row=dict(event_id=f'{out.name}_{phase}_{status}_{utc()}',timestamp_utc=utc(),session_id=session['session_id'],run_id=out.name,
        phase=phase,status=status,scientific_status=scientific,command=command or prov['prepare_command'],output_dir=rel(out),exit_code=exit_code,
        source_base_commit=prov['HEAD'],source_dirty=prov['git_status'],source_manifest_path=rel(out/'SOURCE_MANIFEST.json'),
        source_manifest_sha256=sha(out/'SOURCE_MANIFEST.json'),protocol_path=rel(p),protocol_sha256=sha(p),provenance_path=rel(out/'PROVENANCE.json'),
        artifacts=[rel(x) for x in out.glob('*.json')],blocker_or_failure_reason=reason)
    with (MEM/'RUN_INDEX.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (MEM/'sessions'/f"{session['session_id']}.md").open('a',encoding='utf-8',newline='\n') as f:f.write(f'\n- {utc()} phase={phase}; execution={status}; scientific={scientific}; reason={reason}; command={row["command"]}; exit={exit_code}\n')
    states=read(out/'PHASE_STATUS.json') if (out/'PHASE_STATUS.json').exists() else {}
    states[phase]=dict(status=status,scientific_status=scientific,reason=reason);atomic(out/'PHASE_STATUS.json',states)
    text=f'''# 当前交接：P1-F population 因果判别

更新时间 UTC：{utc()}。分支 codex-refactor；本轮基线 HEAD {prov['HEAD']}。P1-E发布提交bf97a33已确认；本轮P1-F尚未提交/推送。新文件可能被现有allowlist忽略。

五层对象分别记录：p、g、b、beta/u/h、theta。训练只用population目标，teacher用于oracle定义与评价；不据truth选epoch、seed或改阈值。

P1-C/P1-D/P1-D-LR/P1-E均frozen，v1 population=0且BLOCKED保持原样。P1-F把legacy audit与fresh gate分开；不把旧.001/71的数值失败推断为新population失败。

当前阶段：
```json
{json.dumps(states,ensure_ascii=False,indent=2)}
```

当前证据入口：{rel(out)}/PROTOCOL.json、PROVENANCE.json、GATES.json。尚未生成的结果不可视为成功。当前事件run_id={out.name}；执行状态与科学结论分开。

代码manifest：{rel(out)}/SOURCE_MANIFEST.json；完整运行源码在raw/source。历史冻结清单：{session['frozen_manifest']}。与原始结果冲突时核查原始证据。

最小下一步：只执行本protocol允许阶段；基础fresh gate失败则停止训练，endpoint fidelity失败则保留结果、不重跑或解释为结构发现；剂量实验只有预声明paired条件满足才运行0.1倍正则的两条轨迹。

阅读顺序：docs/project_memory/sessions/{session['session_id']}.md → 本轮protocol/provenance → SUMMARY/NOTES（若已存在）→ raw trajectory/states → acawlr_ppp_p1f_population.py。
'''
    tmp=MEM/'CATCH_UP.tmp';tmp.write_text(text,encoding='utf-8',newline='\n');os.replace(tmp,MEM/'CATCH_UP.md')

def protocol(run_id):
    s=read(MEM/'P1F_SESSION.json')
    return dict(stage='P1-F',version=1,run_id=run_id,frozen_at=utc(),task_path=s['prompt_path'],task_sha256=sha(REPO/s['prompt_path']),
      objective='F_lambda=256*(logZ_Q - E_pi_teacher_Q[g])+lambda*R; lambda multiplies all three original explicit L2 terms',
      penalty=dict(beta=.02,u=.2,theta=.001),model='unchanged 675-parameter anchored reduced rank-one',teacher=e.TEACHER_HASH,
      precision=dict(dtype='float64',device='cpu',threads=1,expected_torch='2.12.0+cpu'),
      initialization=dict(standard=dict(archive='mineral_prediction/ACAWLR_PPP_P1C_TS_STATES.pt',case='B_n256_data23_init1011',hash=e.INIT_HASH),
        local=dict(seed=20260918,formula='eta_star + 1e-3*max(1,||eta_star||)*Gaussian_unit_direction; buffers unchanged',perturbation=1e-3),
        note='23/53/71 were data seeds with identical init1011, not independent population starts; exactly two starts, no seed sweep'),
      optimizer=dict(name='Adam',lr=.003,betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,schedule='constant',steps=2000,no_early_stop=True),
      quadrature=dict(train=48,monitor=96,endpoint=192,kind='unchanged mildly nonuniform midpoint cells',refinement='no higher grid or retraining in v1'),
      logging=dict(every_step='48-grid objective/KL/TV/g/b/beta/u/h/q/theta/penalty/block-gradient/actual-block-update',monitor_every=25,
        checkpoints=list(CHECKPOINTS),checkpoint_rule='fixed steps, final2000 only; no best-epoch selection',tail='last200 gradients/total; observations1800,1825,...,2000'),
      gates=dict(legacy='audit only; do not include legacy endpoint in fresh AND-gate',reason='The old empirical endpoint is not used in population target, initial state, controls or objective. Dependency scope changes by user request; numerical thresholds remain unchanged; P1-E gate/result immutable.',
        copy_max=1e-12,teacher_gradient_max=1e-9,KL_identity_atol=1e-10,FD_eps=[1e-4,1e-5,1e-6],FD_atol=1e-7,FD_rtol=1e-4,
        logZ_96_192=1e-4,KL_96_192_atol=1e-6,KL_96_192_rtol=.01,TV_96_192=1e-3,
        teacher_moments_abs=5e-5,teacher_moment_basis='X1,X2,h*X1,h*X2,g; dimensionless; maximum absolute96-192 difference',
        teacher_density_common_grid_TV=1e-3,controls_gradient_per_event=1e-10,controls='global unpenalized and ridge; fixed-h unpenalized and ridge, recomputed on48/96/192',
        endpoint_policy='check each new endpoint; retain failed endpoint as grid_unresolved, no dose if any main endpoint fails; no legacy veto'),
      recovery=dict(density_KL=1e-4,density_TV=.01,predictor_centered_area_and_teacher_RMSE=.02,
        spatial_signal='KL <= .1*delta_global when reliable delta_global>1e-5',field='report independently; not a density gate',
        local_floor=1e-10,local_meaning='if initialKL<10*floor, stability-only; otherwise require >=90% initial excess-KL reduction plus absolute success for local recovery, report final stability separately'),
      definitions=dict(teacher_scale='area-weighted teacher q RMS and (b-beta) Euclidean field RMS on same grid',
        near_zero_q_relative=1e-4,near_zero_field_relative=1e-4,collapse_relative=1e-8,collapse_abs_floor=1e-12,
        collapse_max_factor=10.,collapse='q RMS<=max(1e-12,1e-8*teacher qRMS) AND field contributionRMS<=analogous floor AND max|q|<=10*q floor; global-family degeneration, not necessarily optimal global beta',
        mild_shrink_ratio_max=.8,severe_shrink_ratio_max=.5,degradation_KL='max(1e-4,.1*delta_global)',
        mild='0<q ratio<=.8, not collapsed, acceptable density recovery',severe='q ratio<=.5, not collapsed, KL>=degradation threshold',
        other='shrink_degraded/intermediate/near_zero_without_full_collapse retained; no forced category',
        stable='all9 final monitor observations meet relevant threshold; q ratio range<=.1 and KL range<=.05*delta_global',
        post_growth='report separately whether q ratio ever exceeded .01 before nearzero/collapse; low initial branch is not newly induced collapse',
        stationarity=dict(gradient_L2_per256=1e-3,total_range_per256=1e-5,window=200)),
      stage1=dict(order=['standard_P0','standard_Plambda','local_P0','local_Plambda'],count=4),
      stage2=dict(multiplier=.1,max_extra_trajectories=2,starts=['standard','local'],reuse_lambda0_and1=True,
        trigger='All4 main runs complete2000 and pass endpoint fidelity; at least one matched pair has P0 final density+predictor success AND Plambda stable severe shrinkage or stable collapse over final9 monitor points AND Plambda final density fails. Then exactly .1 runs from both identical saved starts; no other doses or seed expansion.',
        note='Local P0 may test preservation rather than recovering a resolved perturbation; this does not make cold-start success.'),
      causal_scope='Two fixed starts, one teacher and optimizer. Do not infer collapse probability or historical empirical cause from this limited experiment.',
      publication='No automatic commit/push for this new stage')

@torch.no_grad()
def stats(v,w):
    a=w/w.sum();mean=a@v;return dict(mean=float(mean),RMS=float((a*v.square()).sum().sqrt()),variance=float((a*(v-mean).square()).sum()),max_abs=float(v.abs().max()))
@torch.no_grad()
def evaluate_arrays(m,s,t,truth,w):
    a=w/w.sum();X=t['xy']/e.SCALE;u=m.u.detach();beta=m.beta.detach();h=s['h'];q=(X@u)*h;field=h[:,None]*u
    error=s['b']-truth['b'];den=(a[:,None]*truth['b'].square()).sum().sqrt();fieldrms=(a[:,None]*error.square()).sum().sqrt()
    r=e.measures(s['g'],t);r.update(beta_error=float((beta-t['teacher_beta']).norm()),u_norm=float(u.norm()),theta_norm=float(e.vec(m)[4:].norm()),
        h=stats(h,w),q=stats(q,w),field_contribution_RMS=float((a[:,None]*field.square()).sum().sqrt()),b_RMSE=float(fieldrms/math.sqrt(2)),b_relative_RMSE=float(fieldrms/den),
        b_component_RMSE=[float((a*error[:,j].square()).sum().sqrt()) for j in (0,1)],theta_distance=float((e.vec(m)[4:]-t['teacher_theta']).norm()))
    r['q_ratio']=r['q']['RMS']/t['teacher_q_RMS'];r['field_ratio']=r['field_contribution_RMS']/t['teacher_field_RMS']
    canon=e.canonical(m,h,w);r['canonical']=dict(defined=canon['defined'],v=canon['v'])
    if canon['defined']:
        tc=t['teacher_canonical'];v=torch.tensor(canon['v'],dtype=DT);tv=torch.tensor(tc['v'],dtype=DT)
        r['canonical'].update(angle=float(torch.acos((v@tv).clamp(-1,1))),f_RMSE=float((a*(canon['f']-tc['f']).square()).sum().sqrt()))
    return r

def labels(m,scale,delta):
    qfloor=max(1e-12,1e-8*scale['teacher_q_RMS']);bfloor=max(1e-12,1e-8*scale['teacher_field_RMS'])
    collapse=m['q']['RMS']<=qfloor and m['field_contribution_RMS']<=bfloor and m['q']['max_abs']<=10*qfloor
    near=m['q_ratio']<=1e-4 and m['field_ratio']<=1e-4
    density=m['KL']<=1e-4 and m['TV']<=.01;predictor=max(m['centered_g_area_RMSE'],m['centered_g_teacher_RMSE'])<=.02
    category=('collapse' if collapse else 'severe_shrinkage' if m['q_ratio']<=.5 and m['KL']>=max(1e-4,.1*delta) else
              'mild_shrinkage' if m['q_ratio']<=.8 and density else 'shrink_degraded' if m['q_ratio']<=.8 else 'not_shrunken_by_definition')
    return dict(collapse=collapse,near_zero=near,category=category,density_success=density,predictor_success=predictor,
        spatial_signal_success=m['KL']<=.1*delta if delta>1e-5 else None)

def original_inputs():
    archive=tensor_load(ROOT/'ACAWLR_PPP_P1C_TS_STATES.pt');teacher=e.model(archive['teacher_state']).requires_grad_(False)
    initial=archive['cases']['B_n256_data23_init1011']['initial_state'];assert e.state_hash(initial)==e.INIT_HASH
    assert e.state_hash(teacher.state_dict())==e.TEACHER_HASH
    assert all(e.state_hash(archive['cases'][f'B_n256_data{k}_init1011']['initial_state'])==e.INIT_HASH for k in (23,53,71))
    local=e.model(teacher.state_dict());v=e.vec(local);d=torch.randn(v.shape,generator=torch.Generator().manual_seed(20260918),dtype=DT);d/=d.norm();e.setvec(local,v+1e-3*max(1.,float(v.norm()))*d)
    for n,b in teacher.named_buffers():assert torch.equal(b,dict(local.named_buffers())[n])
    return teacher,dict(standard=copy.deepcopy(initial),local=copy.deepcopy(local.state_dict())),d

def grid_targets(teacher):
    grids={}
    for n in (48,96,192):
        xy,w=e.quadrature(n);s=e.surface(teacher,xy);t=e.target(s['g'],w);a=w/w.sum();q=(xy/e.SCALE@teacher.u)*s['h']
        t.update(xy=xy,teacher_beta=teacher.beta.detach(),teacher_theta=e.vec(teacher)[4:],teacher_q_RMS=float((a*q.square()).sum().sqrt()),
                 teacher_field_RMS=float((a[:,None]*(s['h'][:,None]*teacher.u).square()).sum().sqrt()),teacher_canonical=e.canonical(teacher,s['h'],w))
        grids[n]=dict(xy=xy,w=w,truth=s,target=t)
    return grids

def controls(grids):
    out={}
    for n,z in grids.items():
        t=z['target'];X=z['xy']/e.SCALE;h=z['truth']['h'];D=torch.cat([X,h[:,None]*X],1);mu=t['pi']@X;Eg=float(t['pi']@t['g']);row={}
        for name,lam in [('global',0.),('global_ridge',.02)]:
            v=e.solve_global(mu.numpy(),lam);beta=torch.tensor(v['beta'],dtype=DT);logz=math.log(9600)+float(e.global_functions(v['beta'])[0].sum());mass=z['w']*(X@beta-logz).exp()
            v.update(logZ=logz,KL=Eg-float(mu@beta)+logz-float(t['logz']),TV=float((mass-t['pi']).abs().sum()/2));row[name]=v
        for name,lam in [('fixed_h',[0.,0.,0.,0.]),('fixed_h_ridge',[.02,.02,.2,.2])]:row[name]=e.convex_newton(D,t,lam)
        out[str(n)]=row
    return out

def prepare(out):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('raw','raw/source','raw/states','logs','figures'):(out/name).mkdir()
    prot=protocol(out.name);write(out/'PROTOCOL.json',prot)
    session=read(MEM/'P1F_SESSION.json');sources={}
    for path in [*ROOT.glob('*p1f*.py'),ROOT/'acawlr_ppp_p1e_identifiability.py',ROOT/'acawlr_ppp_bridge.py',ROOT/'acawlr_ppp_synthetic.py',ROOT/'acawlr_ppp_teacher_student.py',REPO/'scripts/ACAWLR_improved.py']:
        name=rel(path);sources[name]=sha(path);dest=out/'raw/source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    write(out/'SOURCE_MANIFEST.json',sources)
    prov=dict(start=utc(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),prepare_command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),
        torch=str(torch.__version__),python=sys.version,device='cpu',threads=1,dtype='float64',protocol_sha256=sha(out/'PROTOCOL.json'),source_manifest_sha256=sha(out/'SOURCE_MANIFEST.json'),
        frozen_manifest=session['frozen_manifest'],old_v1_protocol_sha256=sha(OLD/'PROTOCOL.json'),old_v1_gates_sha256=sha(OLD/'GATES.json'))
    write(out/'PROVENANCE.json',prov);memory(out,'prepare','IN_PROGRESS')
    try:
        frozen=frozen_check();assert frozen['passed'],frozen
        teacher,starts,direction=original_inputs();grids=grid_targets(teacher)
        torch.save(dict(teacher=teacher.state_dict(),starts=starts,local_direction=direction),out/'raw/INPUTS.pt')
        for n,g in grids.items():torch.save(g,out/f'raw/GRID_{n}.pt')
        control=controls(grids);write(out/'CONTROLS.json',control)
        # Scope-separated legacy audit is copied/referenced; old frozen files never rewritten.
        legacy=read(OLD/'STRUCTURE.json')['historical']['0.001_71']['fidelity']
        checks={};diagnostics={};checks['environment']=str(torch.__version__)==prot['precision']['expected_torch']
        checks['teacher_hash']=e.state_hash(teacher.state_dict())==e.TEACHER_HASH;checks['standard_hash']=e.state_hash(starts['standard'])==e.INIT_HASH
        checks['legacy_count_zero_verified']=read(OLD/'POPULATION_RESULTS.json')['count']==0
        z96=float(grids[96]['target']['logz']);z192=float(grids[192]['target']['logz']);dlogz=abs(z192-z96)
        common_tv=float((grids[192]['target']['pi']*(math.exp(z192-z96)-1)).abs().sum()/2)
        moments={}
        for n,g in grids.items():
            X=g['xy']/e.SCALE;h=g['truth']['h'];basis=torch.cat([X,h[:,None]*X,g['truth']['g'][:,None]],1);moments[str(n)]=(g['target']['pi']@basis).tolist()
        md=float(np.max(np.abs(np.array(moments['96'])-np.array(moments['192']))));checks['teacher_logZ']=dlogz<=1e-4;checks['teacher_density_common_grid']=common_tv<=1e-3;checks['teacher_moments']=md<=5e-5
        diagnostics['teacher']=dict(logZ96=z96,logZ192=z192,delta_logZ=dlogz,common_grid_TV=common_tv,moments=moments,moment_max_delta=md,
            q_RMS192=grids[192]['target']['teacher_q_RMS'],field_RMS192=grids[192]['target']['teacher_field_RMS'])
        for name in ('global','global_ridge','fixed_h','fixed_h_ridge'):
            fid=e.fidelity(control['96'][name],control['192'][name]);checks['controls_fidelity_'+name]=fid['passed'];diagnostics[name]=fid
            checks['controls_solver_'+name]=all(control[str(n)][name]['converged'] for n in (48,96,192))
        m=e.model(teacher.state_dict());g=grids[48];t=g['target'];s=e.surface(m,g['xy']);copyerrors={k:float((s[k]-g['truth'][k]).abs().max()) for k in ('g','b','h')}
        grad=e.flat_grad(e.pop(m.g(g['xy']),t),m);checks['teacher_copy']=max(copyerrors.values())<=1e-12;checks['teacher_gradient']=float(grad.abs().max())<=1e-9
        diagnostics['implementation']=dict(copy_errors=copyerrors,gradient_max=float(grad.abs().max()))
        initrows={}
        for name,state in starts.items():
            m=e.model(state);rows={}
            for n,g in grids.items():rows[str(n)]=evaluate_arrays(m,e.surface(m,g['xy']),g['target'],g['truth'],g['w'])
            fid=e.fidelity(rows['96'],rows['192']);checks['initial_fidelity_'+name]=fid['passed'];rows['fidelity']=fid;initrows[name]=rows
            gg=e.surface(m,grids[48]['xy'])['g'];tt=grids[48]['target'];gap=float(e.pop(gg,tt)-e.pop(tt['g'],tt));kl=e.measures(gg,tt)['KL'];checks['KL_identity_'+name]=abs(gap-kl)<=1e-10
        write(out/'INITIAL_DIAGNOSTICS.json',initrows)
        m=e.model(starts['standard']);g=grids[48];t=g['target'];d=torch.tensor([.6,-.8],dtype=DT);initial=m.beta.detach().clone();analytic=float(torch.autograd.grad(e.pop(m.g(g['xy']),t),m.beta)[0]@d);fd=[]
        for eps in (1e-4,1e-5,1e-6):
            with torch.no_grad():m.beta.copy_(initial+eps*d)
            hi=float(e.pop(m.g(g['xy']),t).detach())
            with torch.no_grad():m.beta.copy_(initial-eps*d)
            lo=float(e.pop(m.g(g['xy']),t).detach())
            numerical=(hi-lo)/(2*eps);fd.append(dict(epsilon=eps,analytic=analytic,numerical=numerical,passed=abs(numerical-analytic)<=1e-7+1e-4*abs(analytic)))
        checks['gradient_FD']=all(x['passed'] for x in fd);diagnostics['gradient_FD']=fd
        prov.update(teacher_sha256=e.TEACHER_HASH,initial_hashes={k:e.state_hash(v) for k,v in starts.items()},grid_hashes={str(n):e.state_hash(dict(xy=g['xy'],weights=g['w'])) for n,g in grids.items()},
          parameter_order=[dict(name=n,shape=list(p.shape)) for n,p in teacher.named_parameters()],static_completed=utc())
        write(out/'PROVENANCE.json',prov)
        gate=dict(legacy_audit=dict(source=rel(OLD/'STRUCTURE.json')+'#/historical/0.001_71/fidelity',result=legacy,blocks_fresh=False,v1_status='BLOCKED',v1_trajectories=0),
          fresh_pretraining=dict(checks=checks,passed=all(checks.values()),failed=[k for k,v in checks.items() if not v],diagnostics=diagnostics),endpoints={})
        write(out/'GATES.json',gate)
        confirmation=dict(checked_at=utc(),protocol_sha256=sha(out/'PROTOCOL.json'),source_manifest_sha256=sha(out/'SOURCE_MANIFEST.json'),
             protocol_existed_before_first_forward=True,neural_updates_so_far=0,definitions_have_no_results_dependency=True,fresh_gate_passed=gate['fresh_pretraining']['passed'],
             legacy_veto_removed_by_explicit_user_request=True)
        write(out/'PRETRAIN_CONFIRMATION.json',confirmation)
        memory(out,'prepare','COMPLETED' if gate['fresh_pretraining']['passed'] else 'BLOCKED','passed' if gate['fresh_pretraining']['passed'] else 'numerical_unresolved',str(gate['fresh_pretraining']['failed']),exit_code=0)
        print(json.dumps(gate,indent=2),flush=True)
    except BaseException:
        (out/'logs/prepare_error.txt').write_text(traceback.format_exc(),encoding='utf-8');memory(out,'prepare','FAILED',reason='logs/prepare_error.txt');raise
    finally:write(out/'FROZEN_VERIFY_PREPARE.json',frozen_check())

def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        key=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,key))
        elif isinstance(v,(list,tuple)):out[key]=json.dumps(v)
        else:out[key]=v
    return out

def endpoint_summary(label,lam,history,observations,final,fidelity,delta,status):
    tail=[x['96'] for x in observations if x['step']>=1800];qrange=max(x['q_ratio'] for x in tail)-min(x['q_ratio'] for x in tail) if tail else None
    klrange=max(x['KL'] for x in tail)-min(x['KL'] for x in tail) if tail else None
    stable=len(tail)==9 and qrange<=.1 and klrange<=.05*delta
    severe=stable and all(x['labels']['category']=='severe_shrinkage' for x in tail)
    collapse=stable and all(x['labels']['collapse'] for x in tail)
    shrink=stable and all(x['q_ratio']<=.8 for x in tail)
    last=final['192'];gmax=max(x['gradient']['all'] for x in history[-200:])/256;span=(max(x['total_objective'] for x in history[-200:])-min(x['total_objective'] for x in history[-200:]))/256
    initial=observations[0]['192'];resolved=initial['KL']>=1e-9
    growth=any(x['q_ratio']>.01 for x in history)
    return dict(label=label,lambda_multiplier=lam,status=status,last_step=history[-1]['step'],initial192=initial,final=final,fidelity=fidelity,
      density_success=fidelity['passed'] and status=='COMPLETED' and last['labels']['density_success'],predictor_success=fidelity['passed'] and status=='COMPLETED' and last['labels']['predictor_success'],
      stable_shrinkage=shrink,stable_severe_shrinkage=severe,stable_collapse=collapse,tail_q_ratio_range=qrange,tail_KL_range=klrange,
      stationarity=dict(max_gradient_per256=gmax,total_range_per256=span,passed=gmax<=1e-3 and span<=1e-5),
      branch_ever_grew=growth,collapse_observed_steps=[x['step'] for x in observations if x['96']['labels']['collapse']],
      local_probe=dict(initial_gap_resolved=resolved,interpretation='local_recovery_probe' if resolved else 'stability_only',excess_gap_reduction90=last['KL']<=.1*initial['KL']),
      penalty_reference=history[-1]['penalty_reference'],penalty_effective=history[-1]['penalty_effective'],steps=history[-1]['step'])

def trajectory(out,label,state,lam,grids,teacher,delta):
    m=e.model(state);opt=torch.optim.Adam(m.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    grid=grids[48];xy,w,t=grid['xy'],grid['w'],grid['target'];X=xy/e.SCALE;history=[];observations=[];prev=e.vec(m);status='COMPLETED';step=0
    start=time.perf_counter();jsonl=out/'raw'/f'{label}_trajectory.jsonl';csvpath=out/'raw'/f'{label}_trajectory.csv'
    # A trajectory cannot silently overwrite or restart after interruption.
    with jsonl.open('x',encoding='utf-8',newline='\n') as jf,csvpath.open('x',encoding='utf-8',newline='') as cf:
        writer=None
        try:
            for step in range(2001):
                opt.zero_grad(set_to_none=True);h=m.h(xy);b=m.beta+h[:,None]*m.u;g=(X*b).sum(1);data=256*e.pop(g,t);reg=e.PENALTY(m);total=data+lam*reg;total.backward()
                current=e.vec(m);gradient=torch.cat([p.grad.detach().flatten() for p in m.parameters()]);update=current-prev
                if not torch.isfinite(total).all() or not torch.isfinite(gradient).all():raise FloatingPointError(f'nonfinite state/gradient at step{step}')
                metrics=evaluate_arrays(m,dict(h=h.detach(),b=b.detach(),g=g.detach()),t,grid['truth'],w);metrics['labels']=labels(metrics,t,delta)
                parts=e.penalty_parts(m);row=dict(step=step,total_objective=float(total.detach()),population_likelihood=float(data.detach()),likelihood_excess=256*metrics['KL'],
                    penalty_reference=parts,penalty_effective={k:lam*v for k,v in parts.items()},penalty_total=float(lam*reg.detach()),
                    gradient=dict(beta=float(gradient[:2].norm()),u=float(gradient[2:4].norm()),theta=float(gradient[4:].norm()),all=float(gradient.norm())),
                    actual_update=dict(beta=float(update[:2].norm()),u=float(update[2:4].norm()),theta=float(update[4:].norm()),all=float(update.norm())),**metrics)
                history.append(row);jf.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');flat=flatten(row)
                if writer is None:writer=csv.DictWriter(cf,fieldnames=list(flat));writer.writeheader()
                # Canonical fields may become undefined when branch vanishes; use stable columns.
                writer.writerow({k:flat.get(k) for k in writer.fieldnames})
                if step%25==0:
                    obs=dict(step=step)
                    for n in ((48,96,192) if step in (0,2000) else (48,96)):
                        zz=grids[n];v=metrics if n==48 else evaluate_arrays(m,e.surface(m,zz['xy']),zz['target'],zz['truth'],zz['w']);v['labels']=labels(v,zz['target'],delta);obs[str(n)]=v
                    obs['logZ48_96_abs']=abs(obs['48']['logZ']-obs['96']['logZ']);observations.append(obs)
                    atomic(out/'raw'/f'{label}_observations.json',observations);jf.flush();cf.flush()
                if step in CHECKPOINTS:
                    torch.save(dict(step=step,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),initial_sha256=e.state_hash(state),lambda_multiplier=lam),out/'raw/states'/f'{label}_{step:04d}.pt')
                    print(f'{label} step={step} KL48={row["KL"]:.8g} qratio={row["q_ratio"]:.6g} total={row["total_objective"]:.9f}',flush=True)
                if step==2000:break
                prev=current.clone();opt.step()
        except BaseException as err:
            status='FAILED';torch.save(dict(step=step,model=m.state_dict(),optimizer=opt.state_dict()),out/'raw/states'/f'{label}_failure.pt')
            (out/'logs'/f'{label}_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            if not isinstance(err,FloatingPointError):raise
    final={}
    for n in (96,192):
        z=grids[n];s=e.surface(m,z['xy']);v=evaluate_arrays(m,s,z['target'],z['truth'],z['w']);v['labels']=labels(v,z['target'],delta);final[str(n)]=v;torch.save(s,out/'raw'/f'{label}_surface_{n}.pt')
    summary=endpoint_summary(label,lam,history,observations,final,e.fidelity(final['96'],final['192']),delta,status)
    summary['seconds']=time.perf_counter()-start
    # Compare the dense objective to known feasible controls on the same quadrature.
    summary['penalized_excess_over_teacher_192']=256*final['192']['KL']+lam*(sum(summary['penalty_reference'].values())-float(e.PENALTY(teacher)))
    write(out/'raw'/f'{label}_summary.json',summary);return summary

def dose_trigger(results):
    names=('standard_P0','standard_Plambda','local_P0','local_Plambda')
    complete=all(n in results and results[n]['status']=='COMPLETED' and results[n]['steps']==2000 and results[n]['fidelity']['passed'] for n in names)
    qualifying=[]
    if complete:
        for start in ('standard','local'):
            p0,pl=results[start+'_P0'],results[start+'_Plambda']
            if p0['density_success'] and p0['predictor_success'] and not pl['density_success'] and (pl['stable_severe_shrinkage'] or pl['stable_collapse']):qualifying.append(start)
    return dict(eligible=bool(qualifying),qualifying_pairs=qualifying,all_main_complete_and_fidelity=complete,
        reason='predeclared paired criterion met' if qualifying else 'predeclared paired criterion not met; no dose search')

def train(out):
    torch.set_num_threads(1);prot=read(out/'PROTOCOL.json');prov=read(out/'PROVENANCE.json');confirm=read(out/'PRETRAIN_CONFIRMATION.json');gate=read(out/'GATES.json')
    assert sha(out/'PROTOCOL.json')==prov['protocol_sha256']==confirm['protocol_sha256']
    assert all(sha(REPO/p)==h for p,h in read(out/'SOURCE_MANIFEST.json').items()),'source changed since prepare'
    assert frozen_check()['passed']
    if not gate['fresh_pretraining']['passed']:raise RuntimeError('Fresh pretraining gate blocked; no training')
    marker=out/'TRAINING_STARTED.json'
    with marker.open('x',encoding='utf-8') as f:json.dump(dict(time=utc(),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),protocol_sha256=sha(out/'PROTOCOL.json')),f)
    memory(out,'stage1','IN_PROGRESS',command=read(marker)['command']);inputs=tensor_load(out/'raw/INPUTS.pt');teacher=e.model(inputs['teacher']).requires_grad_(False)
    grids={n:tensor_load(out/f'raw/GRID_{n}.pt') for n in (48,96,192)};delta=read(out/'CONTROLS.json')['192']['global']['KL'];results={}
    try:
        for start in ('standard','local'):
            for label,lam in (('P0',0.),('Plambda',1.)):
                name=start+'_'+label;memory(out,name,'IN_PROGRESS');summary=trajectory(out,name,inputs['starts'][start],lam,grids,teacher,delta);results[name]=summary
                gate['endpoints'][name]=summary['fidelity'];atomic(out/'GATES.json',gate);atomic(out/'POPULATION_COMPARISON.json',results)
                memory(out,name,summary['status'],'passed' if summary['fidelity']['passed'] else 'numerical_unresolved',exit_code=0)
        trigger=dose_trigger(results);write(out/'DOSE_DECISION.json',trigger);memory(out,'stage1','COMPLETED','passed' if trigger['all_main_complete_and_fidelity'] else 'numerical_unresolved')
        if trigger['eligible']:
            memory(out,'stage2','IN_PROGRESS',reason='Frozen dose trigger met; exactly two lambda0.1 trajectories')
            for start in ('standard','local'):
                name=start+'_Psmall';summary=trajectory(out,name,inputs['starts'][start],.1,grids,teacher,delta);results[name]=summary;gate['endpoints'][name]=summary['fidelity'];atomic(out/'GATES.json',gate);atomic(out/'POPULATION_COMPARISON.json',results);memory(out,name,summary['status'],'passed' if summary['fidelity']['passed'] else 'numerical_unresolved')
            memory(out,'stage2','COMPLETED','passed' if all(v['fidelity']['passed'] for v in results.values()) else 'numerical_unresolved')
        else:memory(out,'stage2','SKIPPED','not_evaluated',trigger['reason'])
        write(out/'TRAJECTORIES.json',dict(runs={k:dict(jsonl=rel(out/'raw'/f'{k}_trajectory.jsonl'),csv=rel(out/'raw'/f'{k}_trajectory.csv'),observations=rel(out/'raw'/f'{k}_observations.json'),summary=rel(out/'raw'/f'{k}_summary.json')) for k in results},count=len(results)))
        write(out/'TRAINING_END.json',dict(time=utc(),trajectories=len(results),updates=sum(v['steps'] for v in results.values()),exit_code=0))
    except BaseException:
        (out/'logs/train_error.txt').write_text(traceback.format_exc(),encoding='utf-8');memory(out,'training','FAILED',reason='logs/train_error.txt; no automatic restart');raise
    finally:write(out/'FROZEN_VERIFY_TRAIN.json',frozen_check())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','train'),required=True);parser.add_argument('--output-dir',required=True,type=Path);args=parser.parse_args()
    {'prepare':prepare,'train':train}[args.phase](args.output_dir.resolve())
