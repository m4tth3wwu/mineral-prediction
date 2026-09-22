"""Read-only empirical PPP value/gradient comparison on frozen P1-G endpoints."""
import datetime,json,math,subprocess,sys,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import acawlr_ppp_p1f_population as f
from acawlr_ppp_bridge import profiled_ppp,streaming_ppp_value_and_grad
e=f.e;ROOT,REPO,MEM=f.ROOT,f.REPO,f.MEM
OLD=ROOT/'p1g_empirical/20260922T054836Z_p1g_v1';POP=ROOT/'p1f_population/20260917T121240Z_p1f_v1';DENSE=ROOT/'p1g_quadrature/20260922T062242Z_p1g_q192_384'
NAMES=['empirical_P0','empirical_Plambda']
def flat(v):return torch.cat([x.detach().flatten() for x in v])
def comparison(a,b):
    na,nb=float(a.norm()),float(b.norm());dot=float(a@b);cos=dot/(na*nb) if na*nb>0 else None
    return dict(norm48=na,norm384=nb,difference_norm=float((a-b).norm()),relative_difference_to384=float((a-b).norm())/nb if nb else None,cosine=cos,angle_degrees=math.degrees(math.acos(max(-1.,min(1.,cos)))) if cos is not None else None,dense_derivative_along_unit_negative48=-dot/na if na else None,negative48_is_dense_descent=dot>0 if na*nb else None,scope='Euclidean negative gradient, not an Adam update; no finite step performed')
def blocks(a,b):return {k:comparison(a[s],b[s]) for k,s in [('all',slice(None)),('beta',slice(0,2)),('u',slice(2,4)),('theta',slice(4,None))]}
def independent_beta_u(m,events,xy,w):
    with torch.no_grad():
        surface=e.surface(m,xy);ge=surface['g'].tolist();weights=w.tolist();top=max(ge);z=top+math.log(math.fsum(a*math.exp(b-top) for a,b in zip(weights,ge)));p=[a*math.exp(b-z) for a,b in zip(weights,ge)];xx=(xy/e.SCALE).tolist();hh=surface['h'].tolist();xe=(events/e.SCALE).tolist();he=m.h(events).tolist();n=len(events)
    beta=[n*math.fsum(pp*x[j] for pp,x in zip(p,xx))-math.fsum(x[j] for x in xe) for j in [0,1]]
    u=[n*math.fsum(pp*h*x[j] for pp,h,x in zip(p,hh,xx))-math.fsum(h*x[j] for h,x in zip(he,xe)) for j in [0,1]]
    return dict(logZ=z,beta=beta,u=u)
def run(out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    for name in ['raw/source','logs']:(out/name).mkdir(parents=True)
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode('utf-8').strip('\0').split('\0');exclude={'docs/project_memory/'+n for n in ['CATCH_UP.md','DECISIONS.md','RUN_INDEX.jsonl']};frozen={p:f.sha(REPO/p) for p in paths if p not in exclude};f.write(out/'FROZEN_BEFORE.json',frozen)
    sources={f.rel(Path(__file__)):f.sha(__file__)}
    for p in f.read(DENSE/'SOURCE_MANIFEST.json'):sources[p]=f.sha(REPO/p)
    for p in sources:
        dest=out/'raw/source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/p).read_bytes())
    f.write(out/'SOURCE_MANIFEST.json',sources)
    protocol=dict(stage='P1-G fixed-endpoint training-grid gradient audit',run_id=out.name,time=f.utc(),request='User 急须 interpreted as continue the previously proposed48/384 fixed-endpoint gradient check; interpretation stated before execution.',models=NAMES,steps=[2000],grids=[48,384],updates=0,precision='float64 CPU one thread',objective='256 logZ - sum archived data23 g(events)',penalty='also report actual optimized gradient with multiplier0/1; original beta=.02,u=.2,theta=.001',implementation='existing exact two-pass streaming gradient; full-domain logZ, never minibatch normalization',chunk_primary=512,chunk_verification=1024,verification=dict(gradient_atol=1e-8,gradient_rtol=1e-9,value_atol=1e-9,archive_metric_atol=1e-9,independent_beta_u_atol=1e-8),comparison='||g48-g384||/||g384||, cosine, angle, dot-derived directional derivative for all/beta/u/theta, data and actual total separately',interpretation='No significance/path-causality/Adam-step claim. Dense384 is a comparison grid, not established converged-gradient ground truth; value-fidelity does not ensure derivative fidelity. No arbitrary effect-size pass threshold.',scope='Only two frozen endpoints. No training, finite update, extra checkpoint/seed or higher grid.')
    f.write(out/'PROTOCOL.json',protocol)
    prov=dict(time=f.utc(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),torch=str(torch.__version__),python=sys.version,protocol_sha256=f.sha(out/'PROTOCOL.json'),source_manifest_sha256=f.sha(out/'SOURCE_MANIFEST.json'),input_sha256=f.sha(OLD/'raw/INPUTS.pt'),checkpoints={k:dict(path=f.rel(OLD/'raw/states'/f'{k}_2000.pt'),sha256=f.sha(OLD/'raw/states'/f'{k}_2000.pt')) for k in NAMES},grids={str(n):dict(path=f.rel((POP if n==48 else DENSE)/'raw'/f'GRID_{n}.pt'),sha256=f.sha((POP if n==48 else DENSE)/'raw'/f'GRID_{n}.pt')) for n in [48,384]})
    f.write(out/'PROVENANCE.json',prov);checks={};results={}
    try:
        inp=f.tensor_load(OLD/'raw/INPUTS.pt');events=inp['events'];oldprov=f.read(OLD/'PROVENANCE.json');prior=f.read(DENSE/'PROVENANCE.json');cps={k:f.tensor_load(OLD/'raw/states'/f'{k}_2000.pt') for k in NAMES};grids={n:f.tensor_load((POP if n==48 else DENSE)/'raw'/f'GRID_{n}.pt') for n in [48,384]}
        checks.update(input_exact=prov['input_sha256']==oldprov['input_file_sha256'],event_hash=e.state_hash({'events':events})==oldprov['data_sha256'],environment=str(torch.__version__)==oldprov['torch'],checkpoints_exact=all(prov['checkpoints'][k]['sha256']==prior['checkpoint_files'][k]['sha256'] and cp['step']==2000 and all(int(v['step'])==2000 for v in cp['optimizer']['state'].values()) for k,cp in cps.items()))
        f.write(out/'PRECHECKS.json',dict(checks=checks,passed=all(checks.values()),updates=0));assert all(checks.values()),checks
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No updates')),patch.object(torch.optim.SGD,'step',side_effect=AssertionError('No updates')):
            for label,lam in zip(NAMES,[0.,1.]):
                m=e.model(cps[label]['model']);before=e.state_hash(m.state_dict());assert before==prior['model_state_hashes'][label];reg=float(e.PENALTY(m).detach());gp=e.flat_grad(e.PENALTY(m),m).detach();metrics={};vectors={};ver={}
                for n,z in grids.items():
                    xy,w=z['xy'],z['w'];res,gg=streaming_ppp_value_and_grad(m,events,xy,w,1.,512,wrt=tuple(m.parameters()));grad=flat(gg);res2,gg2=streaming_ppp_value_and_grad(m,events,xy,w,1.,1024,wrt=tuple(m.parameters()));g2=flat(gg2)
                    checks[label+f'_chunk_gradient{n}']=torch.allclose(grad,g2,atol=1e-8,rtol=1e-9);checks[label+f'_chunk_value{n}']=abs(float(res.objective-res2.objective))<=1e-9
                    ind=independent_beta_u(m,events,xy,w);ig=torch.tensor(ind['beta']+ind['u'],dtype=torch.float64);checks[label+f'_independent_beta_u{n}']=float((grad[:4]-ig).abs().max())<=1e-8;checks[label+f'_independent_logZ{n}']=abs(float(res.log_z)-ind['logZ'])<=1e-10
                    totalgrad=grad+lam*gp;metrics[str(n)]=dict(empirical_ppp=float(res.objective),logZ=float(res.log_z),penalty_effective=lam*reg,total=float(res.objective)+lam*reg,data_gradient_norm=float(grad.norm()),total_gradient_norm=float(totalgrad.norm()),penalty_gradient_norm=float((lam*gp).norm()))
                    vectors[n]=dict(data=grad,total=totalgrad,penalty=lam*gp);torch.save(dict(parameters=[dict(name=name,shape=list(p.shape)) for name,p in m.named_parameters()],model_hash=before,**vectors[n]),out/'raw'/f'{label}_gradients_{n}.pt')
                    ver[str(n)]=dict(chunk_max_gradient_difference=float((grad-g2).abs().max()),chunk_value_difference=abs(float(res.objective-res2.objective)),independent=ind,independent_beta_u_max_error=float((grad[:4]-ig).abs().max()))
                    if n==48:
                        direct=profiled_ppp(m,events,xy,w,1.);dg=e.flat_grad(direct.objective,m);checks[label+'_direct48_gradient']=torch.allclose(grad,dg,atol=1e-8,rtol=1e-9);checks[label+'_direct48_value']=abs(float(res.objective-direct.objective.detach()))<=1e-9
                        last=json.loads((OLD/'raw'/f'{label}_trajectory.jsonl').read_text(encoding='utf-8').splitlines()[-1]);checks[label+'_archived48']=abs(float(res.objective)-last['empirical_ppp'])<=1e-9 and abs(float(totalgrad.norm())-last['gradient']['all'])<=1e-8
                    else:checks[label+'_archived384_value']=abs(float(res.objective)-f.read(DENSE/'RESULTS.json')[label]['metrics']['384']['empirical_ppp'])<=1e-9
                    print(label,n,json.dumps(metrics[str(n)]),flush=True)
                checks[label+'_state_unchanged']=e.state_hash(m.state_dict())==before
                checks[label+'_regularizer_cancels']=torch.allclose(vectors[48]['total']-vectors[384]['total'],vectors[48]['data']-vectors[384]['data'],atol=1e-10,rtol=1e-10)
                results[label]=dict(metrics=metrics,data_comparison=blocks(vectors[48]['data'],vectors[384]['data']),total_comparison=blocks(vectors[48]['total'],vectors[384]['total']),objective_shift=metrics['384']['empirical_ppp']-metrics['48']['empirical_ppp'],verification=ver,model_hash=before)
                f.write(out/'RESULTS.json',results)
        checks['sources_unchanged']=all(f.sha(REPO/p)==h for p,h in sources.items());checks['protocol_unchanged']=f.sha(out/'PROTOCOL.json')==prov['protocol_sha256'];checks['input_grids_unchanged']=all(f.sha(REPO/v['path'])==v['sha256'] for v in prov['grids'].values());changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h];checks['historical_unchanged']=not changed
        verification=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],frozen_count=len(frozen),changed=changed,updates=0);f.write(out/'FINAL_VERIFY.json',verification)
        f.write(out/'SUMMARY.json',dict(run_id=out.name,updates=0,integrity_passed=verification['passed'],results={k:dict(objective_shift=v['objective_shift'],data=v['data_comparison']['all'],total=v['total_comparison']['all']) for k,v in results.items()},caution='384 derivative convergence not established; no inference about Adam steps or past trajectory causality'))
        assert verification['passed'],verification
    except BaseException:
        (out/'logs/error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:f.write(out/'FROZEN_AFTER.json',dict(count=len(frozen),changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h]))
    print('OUT',f.rel(out),flush=True)
if __name__=='__main__':
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run(ROOT/'p1g_gradient'/f'{stamp}_p1g_grad48_384')
