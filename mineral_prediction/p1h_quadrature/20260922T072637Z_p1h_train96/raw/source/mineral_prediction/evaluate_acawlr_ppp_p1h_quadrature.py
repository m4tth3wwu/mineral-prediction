"""P1-H fixed endpoint evaluation; Adam.step is forbidden in this module."""
import json,math,sys,traceback,inspect,importlib
from pathlib import Path
from unittest.mock import patch
import torch
import acawlr_ppp_p1h_quadrature as h
from check_acawlr_ppp_p1g_quadrature import independent
from check_acawlr_ppp_p1g_gradients import independent_beta_u
from check_acawlr_ppp_p1g_finite_step import values
from acawlr_ppp_bridge import streaming_ppp_value_and_grad
f,e=h.f,h.e;ROOT,REPO=f.ROOT,f.REPO

def flat(v):return torch.cat([x.detach().flatten() for x in v])
def compare(a,b):
    dot=float(a@b);den=float(a.norm()*b.norm());return dict(training_norm=float(a.norm()),dense_norm=float(b.norm()),relative_difference=float((a-b).norm()/b.norm()),angle_degrees=math.degrees(math.acos(max(-1.,min(1.,dot/den)))))
def diagnostics(out,label,cp,events,checks):
    lam=cp['lambda_multiplier'];m=e.model(cp['model']);before=e.state_hash(m.state_dict());gp=e.flat_grad(e.PENALTY(m),m);vectors={};rows={}
    for n in [96,384]:
        z=h.grid(n);r,gs=streaming_ppp_value_and_grad(m,events,z['xy'],z['w'],1.,512,wrt=tuple(m.parameters()));grad=flat(gs);rr,gg=streaming_ppp_value_and_grad(m,events,z['xy'],z['w'],1.,1024,wrt=tuple(m.parameters()))
        checks[f'{label}_chunk{n}']=torch.allclose(grad,flat(gg),atol=1e-8,rtol=1e-9) and abs(float(r.objective-rr.objective))<=1e-9
        ind=independent_beta_u(m,events,z['xy'],z['w']);checks[f'{label}_beta_u{n}']=float((grad[:4]-torch.tensor(ind['beta']+ind['u'],dtype=torch.float64)).abs().max())<=1e-8
        if n==96:
            direct=h.empirical(m,events,z['xy'],z['w']);dg=e.flat_grad(direct,m);checks[label+'_direct96']=torch.allclose(grad,dg,atol=1e-8,rtol=1e-9) and abs(float(direct.detach()-r.objective))<=1e-9
            last=json.loads((out/'raw'/f'{label}_trajectory.jsonl').read_text(encoding='utf-8').splitlines()[-1]);checks[label+'_final_training']=abs(last['empirical_ppp']-float(r.objective))<=1e-9 and abs(last['gradient']['all']-float((grad+lam*gp).norm()))<=1e-8
        vectors[n]=dict(data=grad,penalty=lam*gp,total=grad+lam*gp);torch.save(dict(model_hash=before,parameters=[dict(name=name,shape=list(p.shape)) for name,p in m.named_parameters()],**vectors[n]),out/'raw'/f'{label}_gradients_{n}.pt')
        rows[str(n)]=dict(data=float(r.objective),total=float(r.objective)+lam*float(e.PENALTY(m).detach()),independent_beta_u=ind)
    pg=cp['optimizer']['param_groups'][0];ids=pg['params'];states=cp['optimizer']['state'];checks[label+'_optimizer_recipe']=pg['lr']==.003 and tuple(pg['betas'])==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0 and not pg['amsgrad'] and all(int(states[i]['step'])==2000 for i in ids)
    moment=torch.cat([states[i]['exp_avg'].flatten() for i in ids]);variance=torch.cat([states[i]['exp_avg_sq'].flatten() for i in ids]);gr=vectors[96]['total'];b1,b2=pg['betas'];lr,eps=pg['lr'],pg['eps'];t=2001
    mn=moment.clone().lerp_(gr,1-b1);vn=variance.clone().mul_(b2).addcmul_(gr,gr,value=1-b2);d=(-lr/(1-b1**t))*mn/(vn.sqrt()/math.sqrt(1-b2**t)+eps)
    scalar=[-lr*((b1*a+(1-b1)*g)/(1-b1**t))/(math.sqrt((b2*b+(1-b2)*g*g)/(1-b2**t))+eps) for a,b,g in zip(moment.tolist(),variance.tolist(),gr.tolist())];err=max(abs(a-b) for a,b in zip(d.tolist(),scalar));checks[label+'_candidate_scalar']=err<=1e-14+1e-12*float(d.abs().max())
    torch.save(dict(candidate=d,moment_old=moment,variance_old=variance,moment_candidate=mn,variance_candidate=vn,model_hash=before,candidate_step=2001,optimizer_steps=0),out/'raw'/f'{label}_candidate.pt')
    probe=e.model(cp['model']);theta=e.vec(m);e.setvec(probe,theta+d);effective=e.vec(probe)-theta;checks[label+'_displacement']=float((effective-d).abs().max())<=1e-14
    torch.save(dict(state=probe.state_dict(),base_model_hash=before,effective_displacement=effective,diagnostic_only=True,optimizer_steps=0),out/'raw'/f'{label}_probe_state.pt')
    with torch.no_grad():
        for n in [96,384]:
            z=h.grid(n);base,raw0=values(m,events,z,lam);end,raw1=values(probe,events,z,lam);torch.save(dict(base=raw0,probe=raw1),out/'raw'/f'{label}_probe_scores_{n}.pt');changes={}
            checks[f'{label}_probe_math{n}']=all(abs(v[k]-v['independent'][k])<=1e-9 for v in [base,end] for k in ['data','penalty','total','logZ'])
            checks[f'{label}_probe_baseline{n}']=abs(base['total']-rows[str(n)]['total'])<=1e-9
            for kind in ['data','penalty','total']:
                vec=vectors[n][kind];pred=float(vec@d);actual=end[kind]-base[kind];inddot=math.fsum(a*b for a,b in zip(vec.tolist(),d.tolist()));checks[f'{label}_{kind}_dot{n}']=abs(pred-inddot)<=1e-12*max(1.,abs(pred))
                changes[kind]=dict(prediction=pred,actual=actual,remainder=actual-pred,reversal=pred < -1e-8 and actual >1e-8,blocks={name:float(vec[sl]@d[sl]) for name,sl in [('beta',slice(0,2)),('u',slice(2,4)),('theta',slice(4,None))]})
            rows[str(n)].update(base=base,probe=end,changes=changes)
    checks[label+'_base_unchanged']=before==e.state_hash(m.state_dict())==e.state_hash(cp['model'])
    return dict(grids=rows,gradient_comparison={kind:{name:compare(vectors[96][kind][sl],vectors[384][kind][sl]) for name,sl in [('all',slice(None)),('beta',slice(0,2)),('u',slice(2,4)),('theta',slice(4,None))]} for kind in ['data','total']},candidate_norm=float(d.norm()),scalar_error=err,optimizer_steps=0,temporary_displacements=1)

def run(out):
    torch.set_num_threads(1);assert h.read(out/'TRAINING_END.json')['updates']==4000
    with (out/'EVALUATION_STARTED.json').open('x',encoding='utf-8') as q:json.dump(dict(time=h.utc(),optimizer_steps=0),q)
    sources=[Path(__file__),ROOT/'check_acawlr_ppp_p1g_quadrature.py',ROOT/'check_acawlr_ppp_p1g_gradients.py',ROOT/'check_acawlr_ppp_p1g_finite_step.py'];manifest={f.rel(p):f.sha(p) for p in sources}
    runtime=Path(inspect.getsourcefile(importlib.import_module('torch.optim.adam')));dest=out/'raw/source/runtime/torch_adam.py';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(runtime.read_bytes())
    for p in sources:
        dest=out/'raw/source'/f.rel(p);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    h.write(out/'EVALUATION_SOURCE_MANIFEST.json',manifest);h.write(out/'EVALUATION_PROVENANCE.json',dict(time=h.utc(),runtime_adam_path=str(runtime),runtime_adam_sha256=f.sha(runtime),protocol_sha256=f.sha(out/'PROTOCOL.json'),sources=manifest))
    inp=f.tensor_load(out/'raw/INPUTS.pt');events=inp['events'];checks={};results={};diag={};delta=h.read(h.POP/'CONTROLS.json')['192']['global']['KL'];checkpoint_hashes={}
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No evaluation updates')),patch.object(torch.optim.SGD,'step',side_effect=AssertionError('No evaluation updates')):
        for train_n,folder in [(48,h.OLD),(96,out)]:
            for label,lam in zip(h.NAMES,[0.,1.]):
                key=f'train{train_n}_{label}';path=folder/'raw/states'/f'{label}_2000.pt';checkpoint_hashes[f.rel(path)]=f.sha(path);cp=f.tensor_load(path);m=e.model(cp['model']);before=e.state_hash(m.state_dict());checks[key+'_step2000']=cp['step']==2000 and all(int(v['step'])==2000 for v in cp['optimizer']['state'].values());metrics={};independent_checks={}
                for n in [384,768]:
                    z=h.grid(n);surface=e.surface(m,z['xy']);met=f.evaluate_arrays(m,surface,z['target'],z['truth'],z['w']);met['labels']=f.labels(met,z['target'],delta);met['empirical_ppp']=256*met['logZ']-float(m.g(events).detach().sum());met['penalty_effective']=lam*float(e.PENALTY(m).detach());met['total_objective']=met['empirical_ppp']+met['penalty_effective'];metrics[str(n)]=met
                    torch.save(surface,out/'raw'/f'{key}_surface_{n}.pt');ind=independent(surface,z['truth'],z['w']);checks[key+f'_independent{n}']=max(abs(ind[k]-met[k]) for k in ind)<=1e-10;independent_checks[str(n)]=ind
                    if train_n==48 and n==384:
                        old=h.read(h.DENSE/'RESULTS.json')[label]['metrics']['384'];checks[key+'_archive384']=all(abs(met[k]-old[k])<=1e-12 for k in ['KL','TV','logZ','q_ratio','field_ratio','b_RMSE'])
                    print(key,n,json.dumps({k:met[k] for k in ['KL','q_ratio','field_ratio','b_RMSE']}),flush=True)
                checks[key+'_unchanged']=e.state_hash(m.state_dict())==before;results[key]=dict(training_grid=train_n,lambda_multiplier=lam,model_hash=before,metrics=metrics,fidelity=e.fidelity(metrics['384'],metrics['768']),independent=independent_checks);h.write(out/'ENDPOINTS.json',results)
                if train_n==96:
                    diag[label]=diagnostics(out,label,cp,events,checks);h.write(out/'ENDPOINT_DIAGNOSTICS.json',diag);print(label,'diagnostics completed',flush=True)
    checks['checkpoints_unchanged']=all(f.sha(REPO/p)==v for p,v in checkpoint_hashes.items());checks['sources_unchanged']=all(f.sha(REPO/p)==v for p,v in manifest.items());checks['frozen']=h.frozen_check()['passed'];h.write(out/'EVALUATION_INPUTS.json',checkpoint_hashes)
    gates=h.read(out/'GATES.json');gates['endpoints']={k:v['fidelity'] for k,v in results.items()};h.write(out/'GATES.json',gates);ver=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],optimizer_steps=0,temporary_displacements=2);h.write(out/'EVALUATION_VERIFY.json',ver);h.record(out,'evaluation','COMPLETED',ver);assert ver['passed'],ver
if __name__=='__main__':
    out=REPO/h.session()['output_dir']
    try:run(out)
    except BaseException:
        (out/'logs/evaluation_error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
