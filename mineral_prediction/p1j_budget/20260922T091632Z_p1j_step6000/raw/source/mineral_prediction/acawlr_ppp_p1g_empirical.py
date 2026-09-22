"""P1-G fixed empirical pair. Historical P1-C--F remain read-only."""
import copy,json,math,subprocess,sys,traceback,argparse,time
from pathlib import Path
from collections import deque
import torch
import acawlr_ppp_p1f_population as f
from acawlr_ppp_bridge import profiled_ppp,rank_one_value_and_grad
e=f.e;ROOT=f.ROOT;REPO=f.REPO;MEM=f.MEM
POP=ROOT/'p1f_population/20260917T121240Z_p1f_v1'
read,write,sha,utc,rel=f.read,f.write,f.sha,f.utc,f.rel

def session():return read(MEM/'P1G_SESSION.json')
def frozen_check():
    files=read(REPO/session()['frozen_manifest']);bad=[p for p,h in files.items() if sha(REPO/p)!=h]
    return dict(time=utc(),count=len(files),passed=not bad,changed=bad)
def record(out,phase,status,detail=None):
    s=session();row=dict(timestamp_utc=utc(),session_id=s['session_id'],run_id=out.name,phase=phase,status=status,detail=detail,output_dir=rel(out),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),source_base_commit=s['base_commit'])
    with (MEM/'RUN_INDEX.jsonl').open('a',encoding='utf-8') as q:q.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (MEM/'sessions'/f"{s['session_id']}.md").open('a',encoding='utf-8') as q:q.write('\n'+json.dumps(row,ensure_ascii=False)+'\n')
def empirical(m,events,xy,w):return profiled_ppp(m,events,xy,w,1.).objective
def event_statistics(v):return dict(shape=list(v.shape),dtype=str(v.dtype),mean=v.mean(0).tolist(),std=v.std(0).tolist(),min=v.min(0).values.tolist(),max=v.max(0).values.tolist(),sum=v.sum(0).tolist())
def inputs():
    lr=f.tensor_load(ROOT/'ACAWLR_PPP_P1D_LR_STATES.pt');d=f.tensor_load(ROOT/'ACAWLR_PPP_P1D_OPTIMIZATION_STATES.pt');c=lr['cases']['0.003_23']
    old=next(x for x in read(ROOT/'ACAWLR_PPP_P1D_LR_RESULTS.json')['cases'] if x['learning_rate']==.003 and x['data_seed']==23);pop=f.tensor_load(POP/'raw/INPUTS.pt')
    checks=dict(data_hash=e.state_hash({'events':c['events']})==old['event_sha256'],data_bit_exact_P1D=torch.equal(c['events'],d['cases'][23]['events']),data_statistics=event_statistics(c['events'])==event_statistics(d['cases'][23]['events']),initialization=e.state_hash(c['initial_state'])==old['initial_state_sha256']==e.INIT_HASH==e.state_hash(pop['starts']['standard'])==e.state_hash(d['cases'][23]['initial_state']),teacher=e.state_hash(lr['teacher_state'])==e.state_hash(d['teacher_state'])==e.state_hash(pop['teacher'])==e.TEACHER_HASH,n_dtype=c['events'].shape==(256,2) and c['events'].dtype==torch.float64)
    return c['events'].clone(),copy.deepcopy(c['initial_state']),copy.deepcopy(lr['teacher_state']),checks

def protocol():
    p=read(POP/'PROTOCOL.json');s=session();v={k:copy.deepcopy(p[k]) for k in ['optimizer','precision','quadrature','penalty','recovery','definitions']}
    v.update(stage='P1-G',version=1,frozen_at=utc(),run_id=s['run_id'],task_path=s['prompt_path'],task_sha256=sha(REPO/s['prompt_path']),base_commit=s['base_commit'],population_reference=rel(POP),objective='n*logsumexp(log(w48)+g48)-sum(g(archived events data23)), n=256; historical profiled PPP',trajectories=['empirical_P0','empirical_Plambda'],data_archive='ACAWLR_PPP_P1D_LR_STATES.pt cases[0.003_23], cross-check P1-D cases[23]',initialization=p['initialization']['standard'],teacher=e.TEACHER_HASH,
    pretraining_gates=dict(hash='exact',gradient_atol=1e-10,gradient_rtol=1e-10,objective_atol=1e-10,FD_eps=[1e-4,1e-5,1e-6],FD_atol=1e-6,FD_rtol=1e-4,policy='Any failed pretraining gate: zero updates, no repair/resampling/threshold relaxation'),endpoint_gates=p['gates'],endpoint_policy='Complete paired fixed run then assess both endpoints. Failed fidelity is unresolved, no retraining/refinement; NaN/Inf aborts further updates.',
    logging=dict(every_step='objectives, penalties, population48 metrics, gradients and actual updates',monitor_every=25,checkpoint_every=25,transition_states='previous2/current/next2',gradient_update_alignment='gradient_t generates update_t+1; actual_update_t=parameters_t-parameters_t-1'),
    dynamics=dict(objective_spike='after25: abs(delta total)>1',gradient_spike='after25: grad/256>10 OR grad>10*median(previous25) with >1 absolute increase',transition_candidate='after25: abs(delta qratio)>.1 OR abs(delta fieldratio)>.1 OR collapse indicator change; not a proven basin transition',relative_update='actual L2 update / max(previous parameter L2,1e-30)',objective_conflict='delta total<-1e-8 AND delta empirical>1e-8 AND delta KL>1e-8 AND delta centered_g_area_RMSE>1e-8; step48 and monitor96 separately',sustained_conflict='>=3 consecutive qualifying monitor intervals (>=75 steps)',coarse_conflict='also test fixed checkpoint intervals: every monitored total decreases, net empirical/KL96/g96 increase',overfit='total and empirical decrease while independent KL96 and g96 error increase'),
    interpretation='P1-F collapse indicator retained; joint functional failure additionally requires density and predictor failure. Cases A-F are not exhaustive or mutually exclusive. No collapse != recovery. Interaction=(empirical_lambda-empirical_0)-(population_lambda-population_0); one dataset/init, no significance claim. No positive bias amplification claim without positive empirical degradation contrast.',scope='Exactly two new 2000-update trajectories, lambda0/1. No reruns, model fixes, dose/lr/seed search, best checkpoint or real data.',publication='commit complete evidence and ordinary push codex-refactor after verification')
    return v

def prepare(out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    for n in ['raw/source','raw/states','logs','figures']:(out/n).mkdir(parents=True)
    write(out/'PROTOCOL.json',protocol());record(out,'prepare','IN_PROGRESS')
    try:
        files=[Path(__file__),ROOT/'acawlr_ppp_p1f_population.py',ROOT/'acawlr_ppp_p1e_identifiability.py',ROOT/'acawlr_ppp_bridge.py',ROOT/'acawlr_ppp_synthetic.py',ROOT/'acawlr_ppp_teacher_student.py',REPO/'scripts/ACAWLR_improved.py'];write(out/'SOURCE_MANIFEST.json',{rel(p):sha(p) for p in files})
        for p in files:
            dest=out/'raw/source'/rel(p);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        events,state,teacher,checks=inputs();checks['frozen']=frozen_check()['passed'];checks['environment']=str(torch.__version__)=='2.12.0+cpu' and torch.get_num_threads()==1
        grids={n:f.tensor_load(POP/f'raw/GRID_{n}.pt') for n in (48,96,192)}
        for n,z in grids.items():checks[f'grid{n}']=e.state_hash(dict(xy=z['xy'],weights=z['w']))==read(POP/'PROVENANCE.json')['grid_hashes'][str(n)]
        starts={k:copy.deepcopy(state) for k in ['empirical_P0','empirical_Plambda']};checks['matched_initial_states']=e.state_hash(starts['empirical_P0'])==e.state_hash(starts['empirical_Plambda'])
        torch.save(dict(events=events,starts=starts,teacher=teacher),out/'raw/INPUTS.pt')
        m=e.model(state);z=grids[48];args=(events,z['xy'],z['w']);g0=e.flat_grad(empirical(m,*args),m);g1=e.flat_grad(empirical(m,*args)+e.PENALTY(m),m);gp=e.flat_grad(e.PENALTY(m),m)
        diff=float((g1-g0-gp).abs().max());checks['gradient_difference']=torch.allclose(g1-g0,gp,rtol=1e-10,atol=1e-10)
        ref,reg,total,gr=rank_one_value_and_grad(m,*args,e.PENALTY);checks['legacy_gradient']=torch.allclose(g1,torch.cat([v.flatten() for v in gr]),rtol=1e-10,atol=1e-10)
        manual=len(events)*torch.logsumexp(z['w'].log()+m.g(z['xy']),0)-m.g(events).sum();val=float(empirical(m,*args).detach());checks['finite_sample_formula']=abs(val-float(manual.detach()))<=1e-10
        popval=float((256*e.pop(m.g(z['xy']),z['target'])).detach());checks['not_population']=abs(val-popval)>1e-8
        changed=events.clone();changed[0,0]+=100.;checks['event_dependence']=abs(float(empirical(m,changed,z['xy'],z['w']).detach())-val)>1e-10
        fd=[];v=e.vec(m);direction=torch.zeros_like(v);direction[:2]=torch.tensor([.6,-.8]);analytic=float(g0@direction)
        for eps in (1e-4,1e-5,1e-6):
            e.setvec(m,v+eps*direction);hi=float(empirical(m,*args).detach());e.setvec(m,v-eps*direction);lo=float(empirical(m,*args).detach());num=(hi-lo)/(2*eps);fd.append(dict(epsilon=eps,analytic=analytic,numerical=num,passed=abs(num-analytic)<=1e-6+1e-4*abs(analytic)))
        e.setvec(m,v);checks['finite_difference']=all(x['passed'] for x in fd);init={}
        for n,z in grids.items():init[str(n)]=f.evaluate_arrays(m,e.surface(m,z['xy']),z['target'],z['truth'],z['w'])
        checks['initial_fidelity']=e.fidelity(init['96'],init['192'])['passed'];old=read(POP/'INITIAL_DIAGNOSTICS.json')['standard'];checks['initial_metrics_match_population']=all(abs(init[n][k]-old[n][k])<1e-12 for n in init for k in ['KL','TV','q_ratio','field_ratio','b_RMSE'])
        write(out/'INITIAL_DIAGNOSTICS.json',dict(metrics=init,empirical=val,population=popval,gradient_difference_max=diff,FD=fd,event_statistics=event_statistics(events)))
        prov=dict(time=utc(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),python=sys.version,torch=str(torch.__version__),threads=1,dtype='float64',protocol_sha256=sha(out/'PROTOCOL.json'),source_manifest_sha256=sha(out/'SOURCE_MANIFEST.json'),input_file_sha256=sha(out/'raw/INPUTS.pt'),data_sha256=e.state_hash({'events':events}),initial_sha256=e.state_hash(state),teacher_sha256=e.state_hash(teacher),statistics=event_statistics(events),archive_hashes={rel(ROOT/p):sha(ROOT/p) for p in ['ACAWLR_PPP_P1D_LR_STATES.pt','ACAWLR_PPP_P1D_OPTIMIZATION_STATES.pt']},frozen_manifest=session()['frozen_manifest'],neural_updates_so_far=0)
        write(out/'PROVENANCE.json',prov);gate=dict(pretraining=dict(checks=checks,passed=all(checks.values()),failed=[k for k,v in checks.items() if not v]),endpoints={});write(out/'GATES.json',gate);write(out/'PRETRAIN_CONFIRMATION.json',dict(time=utc(),passed=all(checks.values()),neural_updates=0,protocol_sha256=prov['protocol_sha256']));record(out,'prepare','PASSED' if all(checks.values()) else 'BLOCKED',gate);print(json.dumps(gate),flush=True)
    except BaseException:
        (out/'logs/prepare_error.txt').write_text(traceback.format_exc(),encoding='utf-8');record(out,'prepare','FAILED');raise
    finally:write(out/'FROZEN_VERIFY_PREPARE.json',frozen_check())

def flags(history,row):
    if not history or row['step']<=25:return dict(objective_spike=False,gradient_spike=False,transition_candidate=False,collapse_transition=False,nan_inf=False)
    prev=history[-1];past=sorted(x['gradient']['all'] for x in history[-25:]);med=past[len(past)//2];gr=row['gradient']['all'];col=row['labels']['collapse']!=prev['labels']['collapse']
    return dict(objective_spike=abs(row['total_objective']-prev['total_objective'])>1,gradient_spike=gr/256>10 or (gr>10*max(med,1e-30) and gr-med>1),transition_candidate=abs(row['q_ratio']-prev['q_ratio'])>.1 or abs(row['field_ratio']-prev['field_ratio'])>.1 or col,collapse_transition=col,nan_inf=False)

def trajectory(out,label,state,lam,events,grids,delta):
    m=e.model(state);opt=torch.optim.Adam(m.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False);z=grids[48];xy,w,t=z['xy'],z['w'],z['target'];history=[];obs=[];prev=e.vec(m);recent=deque(maxlen=3);save_until=-1;start=time.perf_counter()
    with (out/'raw'/f'{label}_trajectory.jsonl').open('x',encoding='utf-8') as jf:
        for step in range(2001):
            opt.zero_grad(set_to_none=True);data=empirical(m,events,xy,w);reg=e.PENALTY(m);total=data+lam*reg;total.backward();current=e.vec(m);gr=torch.cat([p.grad.detach().flatten() for p in m.parameters()]);update=current-prev
            if not torch.isfinite(total) or not torch.isfinite(gr).all():
                torch.save(dict(step=step,model=m.state_dict(),optimizer=opt.state_dict()),out/'raw/states'/f'{label}_failure.pt');raise FloatingPointError(f'{label} nonfinite at {step}')
            met=f.evaluate_arrays(m,e.surface(m,xy),t,z['truth'],w);met['labels']=f.labels(met,t,delta);parts=e.penalty_parts(m)
            row=dict(step=step,empirical_ppp=float(data.detach()),total_objective=float(total.detach()),penalty_total=float(lam*reg.detach()),penalty_reference=parts,penalty_effective={k:lam*v for k,v in parts.items()},gradient={k:float(v.norm()) for k,v in [('beta',gr[:2]),('u',gr[2:4]),('theta',gr[4:]),('all',gr)]},actual_update={k:float(v.norm()) for k,v in [('beta',update[:2]),('u',update[2:4]),('theta',update[4:]),('all',update)]},parameter_norm=float(current.norm()),relative_update_norm=float(update.norm()/prev.norm().clamp_min(1e-30)),**met)
            row['flags']=flags(history,row);history.append(row);jf.write(json.dumps(row,allow_nan=False)+'\n');cp=dict(step=step,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),initial_sha256=e.state_hash(state),lambda_multiplier=lam);recent.append(cp)
            if any(row['flags'].values()):
                save_until=step+2
                for old in recent:torch.save(old,out/'raw/states'/f'{label}_{old["step"]:04d}.pt')
            if step%25==0 or step<=save_until:torch.save(cp,out/'raw/states'/f'{label}_{step:04d}.pt')
            if step%25==0:
                v={k:row[k] for k in ['step','empirical_ppp','total_objective','penalty_total']}
                for n in ((48,96,192) if step in (0,2000) else (48,96)):
                    zz=grids[n];mm=met if n==48 else f.evaluate_arrays(m,e.surface(m,zz['xy']),zz['target'],zz['truth'],zz['w']);mm['labels']=f.labels(mm,zz['target'],delta);v[str(n)]=mm;v[f'empirical_ppp_{n}']=256*mm['logZ']-float(m.g(events).detach().sum())
                v['logZ48_96_abs']=abs(v['48']['logZ']-v['96']['logZ']);obs.append(v);f.atomic(out/'raw'/f'{label}_observations.json',obs);jf.flush()
            if step%250==0:print(f'{label} step={step} empirical={row["empirical_ppp"]:.9f} total={row["total_objective"]:.9f} KL48={row["KL"]:.6g} q={row["q_ratio"]:.6g}',flush=True)
            if step==2000:break
            prev=current.clone();opt.step()
    final={n:obs[-1][n] for n in ['96','192']}
    for n in (96,192):torch.save(e.surface(m,grids[n]['xy']),out/'raw'/f'{label}_surface_{n}.pt')
    summary=f.endpoint_summary(label,lam,history,obs,final,e.fidelity(final['96'],final['192']),delta,'COMPLETED');summary.pop('local_probe');summary.update(seconds=time.perf_counter()-start,empirical_ppp=history[-1]['empirical_ppp'],total_objective=history[-1]['total_objective'],empirical_ppp192=obs[-1]['empirical_ppp_192'],training_to_endpoint_logZ=abs(history[-1]['logZ']-final['192']['logZ']),collapse_steps48=[v['step'] for v in history if v['labels']['collapse']],checkpoints=len(list((out/'raw/states').glob(label+'_*.pt'))));write(out/'raw'/f'{label}_summary.json',summary);return summary

def train(out):
    torch.set_num_threads(1);prov=read(out/'PROVENANCE.json');gate=read(out/'GATES.json');assert gate['pretraining']['passed'],'Pretraining blocked'
    assert sha(out/'PROTOCOL.json')==prov['protocol_sha256'] and sha(out/'raw/INPUTS.pt')==prov['input_file_sha256'];assert all(sha(REPO/p)==h for p,h in read(out/'SOURCE_MANIFEST.json').items());assert frozen_check()['passed']
    with (out/'TRAINING_STARTED.json').open('x',encoding='utf-8') as q:json.dump(dict(time=utc(),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv])),q)
    record(out,'training','IN_PROGRESS');v=f.tensor_load(out/'raw/INPUTS.pt');grids={n:f.tensor_load(POP/f'raw/GRID_{n}.pt') for n in (48,96,192)};delta=read(POP/'CONTROLS.json')['192']['global']['KL'];results={}
    try:
        for label,lam in [('empirical_P0',0.),('empirical_Plambda',1.)]:
            results[label]=trajectory(out,label,v['starts'][label],lam,v['events'],grids,delta);write(out/'EMPIRICAL_COMPARISON.json',results)
        gate['endpoints']={k:v['fidelity'] for k,v in results.items()};write(out/'GATES.json',gate);write(out/'TRAJECTORIES.json',dict(count=2,runs={k:dict(steps=2000,jsonl=rel(out/'raw'/f'{k}_trajectory.jsonl'),observations=rel(out/'raw'/f'{k}_observations.json'),checkpoints=v['checkpoints']) for k,v in results.items()}));write(out/'TRAINING_END.json',dict(time=utc(),trajectories=2,updates=4000,exit_code=0));record(out,'training','COMPLETED',dict(endpoint_fidelity=gate['endpoints']))
    except BaseException:
        (out/'logs/train_error.txt').write_text(traceback.format_exc(),encoding='utf-8');record(out,'training','FAILED','No restart');raise
    finally:write(out/'FROZEN_VERIFY_TRAIN.json',frozen_check())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','train'],required=True);a=p.parse_args();{'prepare':prepare,'train':train}[a.phase](REPO/session()['output_dir'])
