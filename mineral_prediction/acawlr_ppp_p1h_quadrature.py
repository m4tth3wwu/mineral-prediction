"""P1-H: fixed train96 pair, against frozen train48; no adaptive runs."""
import copy,json,math,subprocess,sys,traceback,argparse,time
from pathlib import Path
from collections import deque
import torch
import acawlr_ppp_p1g_empirical as g
from acawlr_ppp_bridge import streaming_ppp_value_and_grad
f=g.f;e=f.e;ROOT,REPO,MEM=f.ROOT,f.REPO,f.MEM
read,write,sha,utc,rel=f.read,f.write,f.sha,f.utc,f.rel
POP=g.POP;OLD=ROOT/'p1g_empirical/20260922T054836Z_p1g_v1'
DENSE=ROOT/'p1g_quadrature/20260922T062242Z_p1g_q192_384'
DENSEST=ROOT/'p1g_quadrature/20260922T063312Z_p1g_q384_768'
NAMES=['empirical_P0','empirical_Plambda']
empirical=g.empirical;flags=g.flags

def session():return read(MEM/'P1H_SESSION.json')
def frozen_check():
    files=read(REPO/session()['frozen_manifest']);bad=[p for p,h in files.items() if sha(REPO/p)!=h]
    return dict(count=len(files),passed=not bad,changed=bad)
def record(out,phase,status,detail=None):
    s=session();row=dict(timestamp_utc=utc(),session_id=s['session_id'],run_id=out.name,phase=phase,status=status,detail=detail,output_dir=rel(out),source_base_commit=s['base_commit'])
    for p in [MEM/'RUN_INDEX.jsonl',MEM/'sessions'/ (s['session_id']+'.md')]:
        with p.open('a',encoding='utf-8') as q:q.write(json.dumps(row,ensure_ascii=False)+'\n')
def grid(n):
    if n in [48,96,192]:return f.tensor_load(POP/f'raw/GRID_{n}.pt')
    if n==384:return f.tensor_load(DENSE/'raw/GRID_384.pt')
    z=f.tensor_load(DENSEST/'raw/GRID_768_coordinates.pt');z['truth']=f.tensor_load(DENSEST/'raw/GRID_768_truth.pt');z['target']=f.tensor_load(DENSEST/'raw/GRID_768_target.pt');z['target']['xy']=z['xy'];return z

def prepare(out):
    torch.set_num_threads(1)
    for n in ['raw/source','raw/states','logs','figures']:(out/n).mkdir(parents=True,exist_ok=True)
    assert not (out/'PROTOCOL.json').exists()
    p=read(OLD/'PROTOCOL.json');p.update(stage='P1-H',version=1,frozen_at=utc(),base_commit=session()['base_commit'],run_id=out.name,task_path=session()['prompt_path'],task_sha256=sha(REPO/session()['prompt_path']),objective='256*logsumexp(log(w96)+g96)-sum(g(archived events data23)) + lambda*original penalty',training_grid=96,diagnostic_grid=48,monitor_grids=[48,96,192],evaluation_grids=[384,768],comparison='Only training quadrature changes relative to P1-G; all4 fixed step2000 endpoints evaluated on384/768. Monitor96 common across old/new but is training grid for new; monitor192 independent for new.',endpoint_policy='No adaptive refinement/retraining if384/768 fidelity fails; mark unresolved. NaN aborts further training. No best checkpoint selection.',scope='Exactly2 new trajectories of2000 updates. No optimizer/lr/model/seed changes. Exactly one candidate full displacement on each new endpoint temporary copy; not a2001st training update.')
    p['quadrature']=dict(training=96,per_step_diagnostics=48,monitor_every25=[48,96,192],four_endpoints=[384,768],source='Frozen P1-F and P1-G quadrature archives')
    p['logging'].update(every_step='96 objective/gradients;48 recovery diagnostics; previous96 gradient dot actual update, actual96 loss delta, remainder',monitor_every=25,checkpoint_every=25)
    p['linear_diagnostic']=dict(prediction='gradient(theta[t-1]) dot (theta[t]-theta[t-1])',reversal='prediction < -1e-8 AND actual total delta >1e-8',scope='Available for new runs only; cannot compare historical per-step reversal rates without archived full gradients.')
    p['endpoint_diagnostics']=dict(grids=[96,384],gradient='Exact two-pass streaming, primary512/check1024, direct96 and independent beta/u verification',candidate='Reconstruct Adam next moment from step2000 state and total gradient96; no optimizer.step; one temporary theta+d each. Scalar formula independently verified.',probe='alpha0 and1 only, data/penalty/total96 and384, no recovery claims from probe objective',tolerances=dict(gradient_atol=1e-8,gradient_rtol=1e-9,value_atol=1e-9,candidate_atol=1e-14,candidate_rtol=1e-12),caution='384 derivative convergence not established; finite-step comparison not causal attribution to penalty or universal optimizer failure.')
    write(out/'PROTOCOL.json',p)
    sources={rel(Path(__file__)):sha(__file__)}
    for name in read(OLD/'SOURCE_MANIFEST.json'):sources[name]=sha(REPO/name)
    sources[rel(ROOT/'check_acawlr_ppp_p1g_gradients.py')]=sha(ROOT/'check_acawlr_ppp_p1g_gradients.py')
    for name in sources:
        dest=out/'raw/source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/name).read_bytes())
    write(out/'SOURCE_MANIFEST.json',sources)
    events,state,teacher,checks=g.inputs();checks.update(frozen=frozen_check()['passed'],environment=str(torch.__version__)=='2.12.0+cpu' and torch.get_num_threads()==1)
    grids={n:grid(n) for n in [48,96,192]}
    for n,z in grids.items():checks[f'grid{n}']=e.state_hash(dict(xy=z['xy'],weights=z['w']))==read(POP/'PROVENANCE.json')['grid_hashes'][str(n)]
    m=e.model(state);z=grids[96];v=float(empirical(m,events,z['xy'],z['w']).detach());g0=e.flat_grad(empirical(m,events,z['xy'],z['w']),m);g1=e.flat_grad(empirical(m,events,z['xy'],z['w'])+e.PENALTY(m),m);gp=e.flat_grad(e.PENALTY(m),m)
    checks['regularization_gradient']=torch.allclose(g1-g0,gp,rtol=1e-10,atol=1e-10)
    manual=256*torch.logsumexp(z['w'].log()+m.g(z['xy']),0)-m.g(events).sum();checks['objective_formula']=abs(v-float(manual.detach()))<=1e-10
    sr,sg=streaming_ppp_value_and_grad(m,events,z['xy'],z['w'],1.,512,wrt=tuple(m.parameters()));checks['streaming_gradient']=torch.allclose(g0,torch.cat([a.flatten() for a in sg]),atol=1e-8,rtol=1e-9);checks['streaming_value']=abs(v-float(sr.objective))<=1e-9
    base=e.vec(m);direction=torch.zeros_like(base);direction[:2]=torch.tensor([.6,-.8]);fd=[]
    for eps in [1e-4,1e-5,1e-6]:
        e.setvec(m,base+eps*direction);hi=float(empirical(m,events,z['xy'],z['w']).detach());e.setvec(m,base-eps*direction);lo=float(empirical(m,events,z['xy'],z['w']).detach());num=(hi-lo)/(2*eps);ana=float(g0@direction);fd.append(dict(epsilon=eps,analytic=ana,numerical=num,passed=abs(num-ana)<=1e-6+1e-4*abs(ana)))
    e.setvec(m,base);checks['FD']=all(x['passed'] for x in fd);checks['initial_restored']=e.state_hash(m.state_dict())==e.INIT_HASH
    met={str(n):f.evaluate_arrays(m,e.surface(m,z['xy']),z['target'],z['truth'],z['w']) for n,z in grids.items()};prior=read(OLD/'INITIAL_DIAGNOSTICS.json');checks['initial_metrics']=all(abs(met[n][k]-prior['metrics'][n][k])<1e-12 for n in met for k in ['KL','TV','q_ratio','field_ratio','b_RMSE']);checks['old48_initial_objective']=abs(float(empirical(m,events,grids[48]['xy'],grids[48]['w']).detach())-prior['empirical'])<=1e-10;checks['initial_fidelity']=e.fidelity(met['96'],met['192'])['passed']
    torch.save(dict(events=events,starts={k:copy.deepcopy(state) for k in NAMES},teacher=teacher),out/'raw/INPUTS.pt')
    gridfiles=[POP/f'raw/GRID_{n}.pt' for n in [48,96,192]]+[DENSE/'raw/GRID_384.pt']+[DENSEST/f'raw/GRID_768_{part}.pt' for part in ['coordinates','truth','target']]
    prov=dict(time=utc(),HEAD=session()['base_commit'],python=sys.version,torch=str(torch.__version__),threads=1,protocol_sha256=sha(out/'PROTOCOL.json'),source_manifest_sha256=sha(out/'SOURCE_MANIFEST.json'),input_file_sha256=sha(out/'raw/INPUTS.pt'),initial_sha256=e.state_hash(state),teacher_sha256=e.state_hash(teacher),data_sha256=e.state_hash({'events':events}),grids={rel(p):sha(p) for p in gridfiles},neural_updates_so_far=0)
    write(out/'PROVENANCE.json',prov);write(out/'INITIAL_DIAGNOSTICS.json',dict(metrics=met,empirical96=v,FD=fd));write(out/'GATES.json',dict(pretraining=dict(checks=checks,passed=all(checks.values()),failed=[k for k,v in checks.items() if not v]),endpoints={}));write(out/'PRETRAIN_CONFIRMATION.json',dict(time=utc(),neural_updates=0,passed=all(checks.values()),protocol_sha256=prov['protocol_sha256']));record(out,'prepare','PASSED' if all(checks.values()) else 'BLOCKED',checks);print(json.dumps(checks),flush=True)
def trajectory(out,label,state,lam,events,grids,delta):
    m=e.model(state);opt=torch.optim.Adam(m.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False);z=grids[96];xy,w,t=z['xy'],z['w'],z['target'];history=[];obs=[];prev=e.vec(m);recent=deque(maxlen=3);save_until=-1;start=time.perf_counter();previous_gradient=None
    with (out/'raw'/f'{label}_trajectory.jsonl').open('x',encoding='utf-8') as jf:
        for step in range(2001):
            opt.zero_grad(set_to_none=True);data=empirical(m,events,xy,w);reg=e.PENALTY(m);total=data+lam*reg;total.backward();current=e.vec(m);gr=torch.cat([p.grad.detach().flatten() for p in m.parameters()]);update=current-prev
            if not torch.isfinite(total) or not torch.isfinite(gr).all():
                torch.save(dict(step=step,model=m.state_dict(),optimizer=opt.state_dict()),out/'raw/states'/f'{label}_failure.pt');raise FloatingPointError(f'{label} nonfinite at {step}')
            zz=grids[48];met=f.evaluate_arrays(m,e.surface(m,zz['xy']),zz['target'],zz['truth'],zz['w']);met['labels']=f.labels(met,zz['target'],delta);parts=e.penalty_parts(m)
            row=dict(step=step,empirical_ppp=float(data.detach()),total_objective=float(total.detach()),penalty_total=float(lam*reg.detach()),penalty_reference=parts,penalty_effective={k:lam*v for k,v in parts.items()},gradient={k:float(v.norm()) for k,v in [('beta',gr[:2]),('u',gr[2:4]),('theta',gr[4:]),('all',gr)]},actual_update={k:float(v.norm()) for k,v in [('beta',update[:2]),('u',update[2:4]),('theta',update[4:]),('all',update)]},parameter_norm=float(current.norm()),relative_update_norm=float(update.norm()/prev.norm().clamp_min(1e-30)),**met)
            row['training_grid']=96;row['diagnostic_grid']=48;row['linear']=None if previous_gradient is None else dict(prediction=float(previous_gradient@update),actual=row['total_objective']-history[-1]['total_objective']);
            if row['linear'] is not None:
                v=row['linear'];v['remainder']=v['actual']-v['prediction'];v['reversal']=v['prediction'] < -1e-8 and v['actual'] > 1e-8
            row['flags']=flags(history,row);history.append(row);jf.write(json.dumps(row,allow_nan=False)+'\n');cp=dict(step=step,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),initial_sha256=e.state_hash(state),lambda_multiplier=lam);recent.append(cp)
            if any(row['flags'].values()):
                save_until=step+2
                for old in recent:torch.save(old,out/'raw/states'/f'{label}_{old["step"]:04d}.pt')
            if step%25==0 or step<=save_until:torch.save(cp,out/'raw/states'/f'{label}_{step:04d}.pt')
            if step%25==0:
                v={k:row[k] for k in ['step','empirical_ppp','total_objective','penalty_total']}
                for n in (48,96,192):
                    zz=grids[n];mm=met if n==48 else f.evaluate_arrays(m,e.surface(m,zz['xy']),zz['target'],zz['truth'],zz['w']);mm['labels']=f.labels(mm,zz['target'],delta);v[str(n)]=mm;v[f'empirical_ppp_{n}']=256*mm['logZ']-float(m.g(events).detach().sum())
                v['logZ48_96_abs']=abs(v['48']['logZ']-v['96']['logZ']);obs.append(v);f.atomic(out/'raw'/f'{label}_observations.json',obs);jf.flush()
            if step%250==0:print(f'{label} step={step} empirical={row["empirical_ppp"]:.9f} total={row["total_objective"]:.9f} KL48={row["KL"]:.6g} q={row["q_ratio"]:.6g}',flush=True)
            if step==2000:break
            prev=current.clone();previous_gradient=gr.clone();opt.step()
    summary=dict(label=label,lambda_multiplier=lam,steps=2000,seconds=time.perf_counter()-start,final_objective=history[-1]['total_objective'],checkpoints=len(list((out/'raw/states').glob(label+'_*.pt'))))
    write(out/'raw'/f'{label}_summary.json',summary);return summary

def train(out):
    torch.set_num_threads(1);prov=read(out/'PROVENANCE.json');assert read(out/'GATES.json')['pretraining']['passed']
    assert sha(out/'PROTOCOL.json')==prov['protocol_sha256'] and sha(out/'raw/INPUTS.pt')==prov['input_file_sha256'];assert all(sha(REPO/p)==h for p,h in read(out/'SOURCE_MANIFEST.json').items());assert all(sha(REPO/p)==h for p,h in prov['grids'].items());assert frozen_check()['passed']
    with (out/'TRAINING_STARTED.json').open('x',encoding='utf-8') as q:json.dump(dict(time=utc(),protocol_sha256=prov['protocol_sha256']),q)
    record(out,'training','IN_PROGRESS');v=f.tensor_load(out/'raw/INPUTS.pt');grids={n:grid(n) for n in [48,96,192]};delta=read(POP/'CONTROLS.json')['192']['global']['KL'];results={}
    try:
        for label,lam in zip(NAMES,[0.,1.]):
            results[label]=trajectory(out,label,v['starts'][label],lam,v['events'],grids,delta);write(out/'TRAJECTORIES.json',results)
        write(out/'TRAINING_END.json',dict(time=utc(),trajectories=2,updates=4000,exit_code=0));record(out,'training','COMPLETED',results)
    except BaseException:
        (out/'logs/train_error.txt').write_text(traceback.format_exc(),encoding='utf-8');record(out,'training','FAILED','No restart');raise
    finally:write(out/'FROZEN_VERIFY_TRAIN.json',frozen_check())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','train'],required=True);a=p.parse_args();{'prepare':prepare,'train':train}[a.phase](REPO/session()['output_dir'])
