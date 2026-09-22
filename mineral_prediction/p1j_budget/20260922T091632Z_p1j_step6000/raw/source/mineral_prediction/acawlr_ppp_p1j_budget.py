"""P1-J: exact Adam continuation2000 to6000, no adaptive budget."""
import copy,json,math,subprocess,sys,traceback,argparse,time
from pathlib import Path
from collections import deque
import torch
import acawlr_ppp_p1i_learning_rate as prior
f,e=prior.f,prior.e;ROOT,REPO,MEM=prior.ROOT,prior.REPO,prior.MEM
read,write,sha,utc,rel=prior.read,prior.write,prior.sha,prior.utc,prior.rel
POP=prior.POP;REFERENCE=ROOT/'p1i_learning_rate/20260922T082159Z_p1i_lr001'
START,END=2000,6000;NAMES=prior.NAMES;grid=prior.grid;empirical=prior.empirical;flags=prior.flags

def session():return read(MEM/'P1J_SESSION.json')
def frozen_check():
    files=read(REPO/session()['frozen_manifest']);bad=[p for p,h in files.items() if sha(REPO/p)!=h];return dict(count=len(files),passed=not bad,changed=bad)
def record(out,phase,status,detail=None):
    s=session();row=dict(timestamp_utc=utc(),session_id=s['session_id'],run_id=out.name,phase=phase,status=status,detail=detail,output_dir=rel(out),source_base_commit=s['base_commit'])
    for p in [MEM/'RUN_INDEX.jsonl',MEM/'sessions'/(s['session_id']+'.md')]:
        with p.open('a',encoding='utf-8') as q:q.write(json.dumps(row,ensure_ascii=False)+'\n')
def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def prepare(out):
    torch.set_num_threads(1)
    for n in ['raw/source','raw/states','logs','figures']:(out/n).mkdir(parents=True,exist_ok=True)
    assert not (out/'PROTOCOL.json').exists();p=read(REFERENCE/'PROTOCOL.json');p.update(stage='P1-J',version=1,frozen_at=utc(),base_commit=session()['base_commit'],run_id=out.name,task_path=session()['prompt_path'],task_sha256=sha(REPO/session()['prompt_path']),reference=rel(REFERENCE),comparison='Budget continuation only: same96 objective, lr.001, data/init/teacher/lambda. Reuse each archived step2000 model AND entire Adam state; continue to6000. Fixed stages2000/4000/6000 on384/768. Compare equal2000-update blocks.',scope='Exactly2 continued trajectories,4000 new updates each,8000 total. No reset/retraining/lr/grid/model changes or adaptive extension. Two isolated final6000 candidate probes; not step6001 training.',interpretation='Budget intervention, not an optimization-time equivalence with lr.003x2000. Better empirical fit need not improve teacher recovery. Never call nonstationary residual an irreducible statistical or regularization error.')
    p['optimizer'].update(steps=6000,resume_step=2000,new_updates_per_trajectory=4000,load_complete_adam_state=True)
    p['quadrature']=dict(training=96,per_step_diagnostics=48,monitor_every25=[48,96,192],fixed_evaluation_steps=[2000,4000,6000],evaluation_grids=[384,768],source='Frozen P1-F/P1-G grids')
    p['logging'].update(boundary='Recompute step2000 state metrics/gradient, reuse archived actual-update and linear fields for that already-executed step; equality checked. Inherit original0..1999 history for flags. New first update2001 uses fresh gradient2000 and restored moments.',checkpoint_every=25,mandatory_boundary=[2000,2001],transition_states='previous2/current/next2 within new2000..6000 segment; do not synthesize unavailable pre2000 checkpoints')
    p['linear_diagnostic']['scope']='Compare blocks1..2000,2001..4000,4001..6000; boundary2000 is not a new update. Same96 objective and thresholds.'
    p['endpoint_diagnostics']['candidate']='At6000 only: restore moments and gradient96, construct candidate6001 and one full displacement on temporary copy; zero optimizer steps. Scalar verification.'
    p['resume_gates']=dict(exact='checkpoint hashes, model buffers/parameters, full Adam state including moments and step2000, data/teacher/init',value_atol=1e-9,gradient_atol=1e-8,gradient_rtol=1e-9,candidate_atol=1e-14,candidate_rtol=1e-12,policy='Any failed pre-resume gate -> zero updates; no repairs/restarts/threshold relaxation')
    p['pretraining_gates']=p['resume_gates'];write(out/'PROTOCOL.json',p)
    sources={rel(Path(__file__)):sha(__file__),rel(ROOT/'acawlr_ppp_p1i_learning_rate.py'):sha(ROOT/'acawlr_ppp_p1i_learning_rate.py')};sources.update({k:sha(REPO/k) for k in read(REFERENCE/'SOURCE_MANIFEST.json')})
    for k in sources:
        dest=out/'raw/source'/k;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/k).read_bytes())
    write(out/'SOURCE_MANIFEST.json',sources);inp=f.tensor_load(REFERENCE/'raw/INPUTS.pt');events=inp['events'];rp=read(REFERENCE/'PROVENANCE.json');checks=dict(parent_verified=read(REFERENCE/'FINAL_VERIFY.json')['passed'] and read(REFERENCE/'EVALUATION_VERIFY.json')['passed'],frozen=frozen_check()['passed'],environment=str(torch.__version__)==rp['torch'] and torch.get_num_threads()==1,data=e.state_hash({'events':events})==rp['data_sha256'],teacher=e.state_hash(inp['teacher'])==e.TEACHER_HASH,initial=all(e.state_hash(x)==e.INIT_HASH for x in inp['starts'].values()),grids=all(sha(REPO/k)==v for k,v in rp['grids'].items()))
    resumes={};details={};inputs={rel(REFERENCE/'raw/INPUTS.pt'):sha(REFERENCE/'raw/INPUTS.pt')};z=grid(96)
    for label,lam in zip(NAMES,[0.,1.]):
        path=REFERENCE/'raw/states'/f'{label}_2000.pt';inputs[rel(path)]=sha(path);cp=f.tensor_load(path);resumes[label]=cp;m=e.model(cp['model']);opt=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False);opt.load_state_dict(copy.deepcopy(cp['optimizer']));pg=opt.param_groups[0];checks[label+'_model_restore']=e.state_hash(m.state_dict())==e.state_hash(cp['model']);checks[label+'_optimizer_restore']=equal(opt.state_dict(),cp['optimizer']);checks[label+'_recipe']=cp['step']==2000 and cp['lambda_multiplier']==lam and cp['initial_sha256']==e.INIT_HASH and pg['lr']==.001 and tuple(pg['betas'])==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0 and not pg['amsgrad'] and all(int(v['step'])==2000 for v in cp['optimizer']['state'].values())
        saved=[json.loads(x) for x in (REFERENCE/'raw'/f'{label}_trajectory.jsonl').read_text(encoding='utf-8').splitlines()];inputs[rel(REFERENCE/'raw'/f'{label}_trajectory.jsonl')]=sha(REFERENCE/'raw'/f'{label}_trajectory.jsonl');data=empirical(m,events,z['xy'],z['w']);total=data+lam*e.PENALTY(m);gr=e.flat_grad(total,m);gv=f.tensor_load(REFERENCE/'raw'/f'{label}_gradients_96.pt')['total'];checks[label+'_objective']=abs(float(total.detach())-saved[-1]['total_objective'])<=1e-9;checks[label+'_gradient']=torch.allclose(gr,gv,atol=1e-8,rtol=1e-9) and abs(float(gr.norm())-saved[-1]['gradient']['all'])<=1e-8
        met=f.evaluate_arrays(m,e.surface(m,z['xy']),z['target'],z['truth'],z['w']);old=read(REFERENCE/'raw'/f'{label}_observations.json')[-1]['96'];checks[label+'_metrics']=all(abs(met[k]-old[k])<=1e-12 for k in ['KL','TV','q_ratio','field_ratio','b_RMSE','logZ'])
        states=cp['optimizer']['state'];ids=cp['optimizer']['param_groups'][0]['params'];mom=torch.cat([states[i]['exp_avg'].flatten() for i in ids]);var=torch.cat([states[i]['exp_avg_sq'].flatten() for i in ids]);b1,b2=pg['betas'];t=2001;d=torch.tensor([-pg['lr']*((b1*a+(1-b1)*g)/(1-b1**t))/(math.sqrt((b2*b+(1-b2)*g*g)/(1-b2**t))+pg['eps']) for a,b,g in zip(mom.tolist(),var.tolist(),gr.tolist())],dtype=torch.float64);candidate=f.tensor_load(REFERENCE/'raw'/f'{label}_candidate.pt')['candidate'];error=float((candidate-d).abs().max());checks[label+'_next_candidate']=error<=1e-14+1e-12*float(d.abs().max());details[label]=dict(model_sha256=e.state_hash(cp['model']),candidate_max_error=error,total=float(total.detach()),gradient_norm=float(gr.norm()))
    torch.save(dict(events=events,teacher=inp['teacher'],initial_states=inp['starts'],resumes=resumes),out/'raw/INPUTS.pt');prov=dict(time=utc(),HEAD=session()['base_commit'],python=sys.version,torch=str(torch.__version__),threads=1,protocol_sha256=sha(out/'PROTOCOL.json'),source_manifest_sha256=sha(out/'SOURCE_MANIFEST.json'),input_file_sha256=sha(out/'raw/INPUTS.pt'),data_sha256=rp['data_sha256'],teacher_sha256=e.TEACHER_HASH,initial_sha256=e.INIT_HASH,grids=rp['grids'],resume_inputs=inputs,resume_models=details,new_updates_so_far=0)
    write(out/'PROVENANCE.json',prov);gate=dict(checks=checks,passed=all(checks.values()),failed=[k for k,v in checks.items() if not v]);write(out/'GATES.json',dict(pretraining=gate,endpoints={}));write(out/'PRETRAIN_CONFIRMATION.json',dict(time=utc(),neural_updates=0,passed=gate['passed'],protocol_sha256=prov['protocol_sha256']));write(out/'RESUME_DIAGNOSTICS.json',details);record(out,'prepare','PASSED' if gate['passed'] else 'BLOCKED',gate);print(json.dumps(gate),flush=True)
def trajectory(out,label,resume,lam,events,grids,delta):
    state=resume['model'];saved_history=[json.loads(x) for x in (REFERENCE/'raw'/f'{label}_trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
    m=e.model(state);opt=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False);opt.load_state_dict(copy.deepcopy(resume['optimizer']));z=grids[96];xy,w,t=z['xy'],z['w'],z['target'];history=copy.deepcopy(saved_history[:START]);obs=[];prev=e.vec(m);recent=deque(maxlen=3);save_until=-1;start=time.perf_counter();previous_gradient=None
    with (out/'raw'/f'{label}_trajectory.jsonl').open('x',encoding='utf-8') as jf:
        for step in range(START,END+1):
            opt.zero_grad(set_to_none=True);data=empirical(m,events,xy,w);reg=e.PENALTY(m);total=data+lam*reg;total.backward();current=e.vec(m);gr=torch.cat([p.grad.detach().flatten() for p in m.parameters()]);update=current-prev
            if not torch.isfinite(total) or not torch.isfinite(gr).all():
                torch.save(dict(step=step,model=m.state_dict(),optimizer=opt.state_dict()),out/'raw/states'/f'{label}_failure.pt');raise FloatingPointError(f'{label} nonfinite at {step}')
            zz=grids[48];met=f.evaluate_arrays(m,e.surface(m,zz['xy']),zz['target'],zz['truth'],zz['w']);met['labels']=f.labels(met,zz['target'],delta);parts=e.penalty_parts(m)
            row=dict(step=step,empirical_ppp=float(data.detach()),total_objective=float(total.detach()),penalty_total=float(lam*reg.detach()),penalty_reference=parts,penalty_effective={k:lam*v for k,v in parts.items()},gradient={k:float(v.norm()) for k,v in [('beta',gr[:2]),('u',gr[2:4]),('theta',gr[4:]),('all',gr)]},actual_update={k:float(v.norm()) for k,v in [('beta',update[:2]),('u',update[2:4]),('theta',update[4:]),('all',update)]},parameter_norm=float(current.norm()),relative_update_norm=float(update.norm()/prev.norm().clamp_min(1e-30)),**met)
            row['training_grid']=96;row['diagnostic_grid']=48;row['linear']=None if previous_gradient is None else dict(prediction=float(previous_gradient@update),actual=row['total_objective']-history[-1]['total_objective']);
            if row['linear'] is not None:
                v=row['linear'];v['remainder']=v['actual']-v['prediction'];v['reversal']=v['prediction'] < -1e-8 and v['actual'] > 1e-8
            if step==START:
                for k in ['actual_update','relative_update_norm','linear']:row[k]=copy.deepcopy(saved_history[-1][k])
            row['flags']=flags(history,row);assert step!=START or row==saved_history[-1];history.append(row);jf.write(json.dumps(row,allow_nan=False)+'\n');cp=dict(step=step,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),initial_sha256=e.INIT_HASH,lambda_multiplier=lam,resume_sha256=e.state_hash(state));recent.append(cp)
            if any(row['flags'].values()):
                save_until=step+2
                for old in recent:torch.save(old,out/'raw/states'/f'{label}_{old["step"]:04d}.pt')
            if step%25==0 or step==START+1 or step<=save_until:torch.save(cp,out/'raw/states'/f'{label}_{step:04d}.pt')
            if step%25==0:
                v={k:row[k] for k in ['step','empirical_ppp','total_objective','penalty_total']}
                for n in (48,96,192):
                    zz=grids[n];mm=met if n==48 else f.evaluate_arrays(m,e.surface(m,zz['xy']),zz['target'],zz['truth'],zz['w']);mm['labels']=f.labels(mm,zz['target'],delta);v[str(n)]=mm;v[f'empirical_ppp_{n}']=256*mm['logZ']-float(m.g(events).detach().sum())
                v['logZ48_96_abs']=abs(v['48']['logZ']-v['96']['logZ']);obs.append(v);f.atomic(out/'raw'/f'{label}_observations.json',obs);jf.flush()
            if step%250==0:print(f'{label} step={step} empirical={row["empirical_ppp"]:.9f} total={row["total_objective"]:.9f} KL48={row["KL"]:.6g} q={row["q_ratio"]:.6g}',flush=True)
            if step==END:break
            prev=current.clone();previous_gradient=gr.clone();opt.step()
    summary=dict(label=label,lambda_multiplier=lam,steps=END,new_updates=END-START,seconds=time.perf_counter()-start,final_objective=history[-1]['total_objective'],checkpoints=len(list((out/'raw/states').glob(label+'_*.pt'))))
    write(out/'raw'/f'{label}_summary.json',summary);return summary

def train(out):
    torch.set_num_threads(1);prov=read(out/'PROVENANCE.json');assert read(out/'GATES.json')['pretraining']['passed'];assert sha(out/'PROTOCOL.json')==prov['protocol_sha256'] and sha(out/'raw/INPUTS.pt')==prov['input_file_sha256'];assert all(sha(REPO/k)==v for k,v in read(out/'SOURCE_MANIFEST.json').items());assert all(sha(REPO/k)==v for k,v in prov['grids'].items());assert all(sha(REPO/k)==v for k,v in prov['resume_inputs'].items());assert frozen_check()['passed']
    with (out/'TRAINING_STARTED.json').open('x',encoding='utf-8') as q:json.dump(dict(time=utc(),protocol_sha256=prov['protocol_sha256'],start_step=START,end_step=END),q)
    record(out,'training','IN_PROGRESS');v=f.tensor_load(out/'raw/INPUTS.pt');grids={n:grid(n) for n in [48,96,192]};delta=read(POP/'CONTROLS.json')['192']['global']['KL'];results={}
    try:
        for label,lam in zip(NAMES,[0.,1.]):
            results[label]=trajectory(out,label,v['resumes'][label],lam,v['events'],grids,delta);write(out/'TRAJECTORIES.json',results)
        write(out/'TRAINING_END.json',dict(time=utc(),trajectories=2,updates=8000,start_step=START,end_step=END,exit_code=0));record(out,'training','COMPLETED',results)
    except BaseException:
        (out/'logs/train_error.txt').write_text(traceback.format_exc(),encoding='utf-8');record(out,'training','FAILED','No restart');raise
    finally:write(out/'FROZEN_VERIFY_TRAIN.json',frozen_check())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','train'],required=True);a=p.parse_args();{'prepare':prepare,'train':train}[a.phase](REPO/session()['output_dir'])
