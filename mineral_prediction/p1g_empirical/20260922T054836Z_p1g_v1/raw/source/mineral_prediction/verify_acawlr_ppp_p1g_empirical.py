"""P1-G independent artifact verification; no optimizer updates, no NumPy BLAS."""
import json,math,sys
from pathlib import Path
import torch
import acawlr_ppp_p1g_empirical as g
f,e=g.f,g.e

def verify(out):
    torch.set_num_threads(1);checks={};details={};prov=g.read(out/'PROVENANCE.json');results=g.read(out/'EMPIRICAL_COMPARISON.json');inputs=f.tensor_load(out/'raw/INPUTS.pt');events=inputs['events']
    checks['frozen']=g.frozen_check()['passed'];checks['source_frozen']=all(g.sha(g.REPO/p)==h for p,h in g.read(out/'SOURCE_MANIFEST.json').items());checks['protocol_frozen']=g.sha(out/'PROTOCOL.json')==prov['protocol_sha256'];checks['inputs_frozen']=g.sha(out/'raw/INPUTS.pt')==prov['input_file_sha256']
    checks['exactly_two_runs']=set(results)=={'empirical_P0','empirical_Plambda'} and g.read(out/'TRAINING_END.json')['updates']==4000
    checks['matched_states']=e.state_hash(inputs['starts']['empirical_P0'])==e.state_hash(inputs['starts']['empirical_Plambda'])==e.INIT_HASH
    for label,lam in [('empirical_P0',0.),('empirical_Plambda',1.)]:
        h=[json.loads(x) for x in (out/'raw'/f'{label}_trajectory.jsonl').read_text().splitlines()];obs=g.read(out/'raw'/f'{label}_observations.json');checks[label+'_step_sequence']=[r['step'] for r in h]==list(range(2001));checks[label+'_monitor_sequence']=[r['step'] for r in obs]==list(range(0,2001,25))
        checks[label+'_loss_accounting']=max(abs(r['total_objective']-r['empirical_ppp']-r['penalty_total']) for r in h)<1e-10 and max(abs(r['penalty_total']-lam*sum(r['penalty_reference'].values())) for r in h)<1e-10
        cp_paths={int(p.stem.rsplit('_',1)[1]):p for p in (out/'raw/states').glob(label+'_*.pt')};required=set(range(0,2001,25))
        for row in h:
            if any(row['flags'].values()):required.update(range(max(0,row['step']-2),min(2000,row['step']+2)+1))
        checks[label+'_transition_checkpoints']=required<=set(cp_paths)
        max_update_error=0.;last=None;laststep=None;stateok=True
        for step,path in sorted(cp_paths.items()):
            cp=f.tensor_load(path);opt=cp['optimizer'];pg=opt['param_groups'][0]
            stateok &= cp['step']==step and cp['initial_sha256']==e.INIT_HASH and cp['lambda_multiplier']==lam and pg['lr']==.003 and tuple(pg['betas'])==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0 and not pg['amsgrad']
            stateok &= (len(opt['state'])==0 if step==0 else all(int(v['step'])==step for v in opt['state'].values()))
            m=e.model(cp['model']);v=e.vec(m)
            if laststep is not None and step==laststep+1:max_update_error=max(max_update_error,abs(float((v-last).norm())-h[step]['actual_update']['all']))
            last=v;laststep=step
        checks[label+'_optimizer_states']=bool(stateok);checks[label+'_actual_updates']=max_update_error<1e-12
        cp0=f.tensor_load(cp_paths[0]);checks[label+'_initial_archive_exact']=e.state_hash(cp0['model'])==e.INIT_HASH
        cp=f.tensor_load(cp_paths[2000]);m=e.model(cp['model']);z48=f.tensor_load(g.POP/'raw/GRID_48.pt')
        from acawlr_ppp_bridge import RankOnePenalty,rank_one_value_and_grad
        penalty=RankOnePenalty(beta=.02*lam,u=.2*lam,theta=.001*lam)
        data,pen,total,gr=rank_one_value_and_grad(m,events,z48['xy'],z48['w'],penalty);gradient=torch.cat([x.flatten() for x in gr]);gradient_error=abs(float(gradient.norm())-h[-1]['gradient']['all'])
        checks[label+'_legacy_endpoint_objective_gradient']=abs(float(data.objective.detach())-h[-1]['empirical_ppp'])<1e-9 and abs(float(total)-h[-1]['total_objective'])<1e-9 and gradient_error<1e-8
        enderrors={}
        for n in (96,192):
            z=f.tensor_load(g.POP/f'raw/GRID_{n}.pt');surface=f.tensor_load(out/'raw'/f'{label}_surface_{n}.pt');fresh=e.surface(m,z['xy']);checks[label+f'_surface{n}']=all(torch.equal(surface[k],fresh[k]) for k in surface)
            w=z['w'].tolist();a=[x/math.fsum(w) for x in w];truth=z['truth'];gt=truth['g'].tolist();gs=surface['g'].tolist()
            def logz(scores):
                top=max(scores);return top+math.log(math.fsum(ww*math.exp(v-top) for ww,v in zip(w,scores)))
            zs,zt=logz(gs),logz(gt);pt=[ww*math.exp(v-zt) for ww,v in zip(w,gt)];ps=[ww*math.exp(v-zs) for ww,v in zip(w,gs)]
            diff=[v-t for v,t in zip(gs,gt)];am=math.fsum(aa*d for aa,d in zip(a,diff));tm=math.fsum(p*d for p,d in zip(pt,diff))
            b=surface['b'].tolist();bt=truth['b'].tolist();hh=surface['h'].tolist();xy=z['xy'].tolist();u=m.u.detach().tolist();beta=m.beta.detach().tolist()
            q=[(p[0]/60000*u[0]+p[1]/40000*u[1])*h0 for p,h0 in zip(xy,hh)];qr=math.sqrt(math.fsum(aa*x*x for aa,x in zip(a,q)));fr=math.sqrt(math.fsum(aa*h0*h0*math.fsum(x*x for x in u) for aa,h0 in zip(a,hh)))
            vals=dict(logZ=zs,KL=-math.fsum(p*d for p,d in zip(pt,diff))+zs-zt,TV=math.fsum(abs(p-pp) for p,pp in zip(pt,ps))/2,centered_g_area_RMSE=math.sqrt(math.fsum(aa*(d-am)**2 for aa,d in zip(a,diff))),centered_g_teacher_RMSE=math.sqrt(math.fsum(p*(d-tm)**2 for p,d in zip(pt,diff))),b_RMSE=math.sqrt(math.fsum(aa*math.fsum((v-t)**2 for v,t in zip(bb,tt)) for aa,bb,tt in zip(a,b,bt))/2),q_ratio=qr/z['target']['teacher_q_RMS'],field_ratio=fr/z['target']['teacher_field_RMS'])
            errors={k:abs(v-results[label]['final'][str(n)][k]) for k,v in vals.items()};checks[label+f'_independent_math{n}']=max(errors.values())<1e-10;enderrors[str(n)]=dict(values=vals,max_error=max(errors.values()))
            if n==192:
                empirical192=256*zs-math.fsum(m.g(events).detach().tolist());checks[label+'_empirical192']=abs(empirical192-results[label]['empirical_ppp192'])<1e-9
        details[label]=dict(checkpoint_count=len(cp_paths),max_actual_update_error=max_update_error,endpoint_gradient_error=gradient_error,independent_endpoints=enderrors)
    summary=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],details=details,frozen=g.frozen_check(),note='Execution/integrity verification is separate from endpoint fidelity and scientific recovery gates. No optimizer updates. Independent math.fsum/log/exp reductions avoid the P1-F postprocessing OpenMP conflict.')
    g.write(out/'FINAL_VERIFY.json',summary);g.record(out,'verification','PASSED' if summary['passed'] else 'FAILED',dict(failed=summary['failed']));print(json.dumps(summary,indent=2));return summary['passed']
if __name__=='__main__':sys.exit(0 if verify(g.REPO/g.session()['output_dir']) else 1)
