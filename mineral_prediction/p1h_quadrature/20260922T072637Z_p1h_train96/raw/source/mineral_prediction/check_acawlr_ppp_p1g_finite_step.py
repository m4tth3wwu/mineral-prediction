"""One full candidate displacement on temporary model copies; no optimizer/training."""
import datetime,json,math,subprocess,sys,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import acawlr_ppp_p1f_population as f
e=f.e;ROOT,REPO,MEM=f.ROOT,f.REPO,f.MEM
OLD=ROOT/'p1g_empirical/20260922T054836Z_p1g_v1';GRAD=ROOT/'p1g_gradient/20260922T065517Z_p1g_grad48_384';ADAM=ROOT/'p1g_adam_direction/20260922T070225Z_p1g_adam_candidate';NAMES=['empirical_P0','empirical_Plambda']

def classify(x):return 'decrease' if x < -1e-8 else 'increase' if x>1e-8 else 'numerically_neutral'
def values(model,events,grid,lam):
    scores=e.surface(model,grid['xy'])['g'];es=model.g(events);logz=torch.logsumexp(grid['w'].log()+scores,0);data=256*logz-es.sum();pen=lam*e.PENALTY(model)
    sv=scores.tolist();wv=grid['w'].tolist();ev=es.tolist();top=max(sv);indz=top+math.log(math.fsum(w*math.exp(g-top) for w,g in zip(wv,sv)));inddata=256*indz-math.fsum(ev)
    indpen=lam*math.fsum((.01 if name=='beta' else .1 if name=='u' else .0005)*math.fsum(v*v for v in p.detach().flatten().tolist()) for name,p in model.named_parameters())
    return dict(data=float(data),penalty=float(pen),total=float(data+pen),logZ=float(logz),independent=dict(data=inddata,penalty=indpen,total=inddata+indpen,logZ=indz)),dict(quadrature_scores=scores,event_scores=es)

def run(out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    for n in ['raw/source','logs']:(out/n).mkdir(parents=True)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode('utf-8').strip('\0').split('\0');exclude={'docs/project_memory/'+n for n in ['CATCH_UP.md','DECISIONS.md','RUN_INDEX.jsonl']};frozen={p:f.sha(REPO/p) for p in tracked if p not in exclude};f.write(out/'FROZEN_BEFORE.json',frozen)
    sources={f.rel(Path(__file__)):f.sha(__file__)}
    for p in f.read(ADAM/'SOURCE_MANIFEST.json'):sources[p]=f.sha(REPO/p)
    for p in sources:
        dest=out/'raw/source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/p).read_bytes())
    f.write(out/'SOURCE_MANIFEST.json',sources)
    protocol=dict(stage='P1-G single candidate finite-displacement diagnostic',run_id=out.name,time=f.utc(),request='User approved executing item1 only: evaluate full saved candidate displacement on temporary copies, original archive unchanged, no training/grid experiment.',models=NAMES,base_step=2000,alphas=[0,1],grids=[48,384],candidate_source=f.rel(ADAM),scope='Exactly one temporary parameter displacement per model, no optimizer step/backward/training continuation, no alpha scan or checkpoint selection; not appended as step2001 training evidence.',predictions='saved g dot d for data/penalty/total; actual J(theta+d)-J(theta); residual=actual-predicted; reversal if predicted<-1e-8 and actual>1e-8',precision='float64 CPU1thread',validation=dict(objective_atol=1e-9,gradient_dot_atol=1e-10,parameter_displacement_atol=1e-14,sign_atol=1e-8),interpretation='Negative directional derivative need not guarantee finite-step decrease. Residual measures nonlinearity along one displacement, not a Hessian or proof of curvature vs nonsmoothness.384 is a comparison grid; candidate-point dense fidelity not independently refined.')
    f.write(out/'PROTOCOL.json',protocol);ap=f.read(ADAM/'PROVENANCE.json');gp=f.read(GRAD/'PROVENANCE.json')
    inputs={ap['checkpoints'][k]['path']:ap['checkpoints'][k]['sha256'] for k in NAMES}
    for k in NAMES:inputs[f.rel(ADAM/'raw'/f'{k}_candidate.pt')]=f.sha(ADAM/'raw'/f'{k}_candidate.pt')
    for v in gp['grids'].values():inputs[v['path']]=v['sha256']
    for k in NAMES:
        for v in ap['gradient_files'][k].values():inputs[v['path']]=v['sha256']
    inputs[f.rel(OLD/'raw/INPUTS.pt')]=f.read(OLD/'PROVENANCE.json')['input_file_sha256']
    prov=dict(time=f.utc(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),torch=str(torch.__version__),python=sys.version,protocol_sha256=f.sha(out/'PROTOCOL.json'),source_manifest_sha256=f.sha(out/'SOURCE_MANIFEST.json'),inputs=inputs)
    f.write(out/'PROVENANCE.json',prov);checks={};results={}
    try:
        checks['input_hashes']=all(f.sha(REPO/p)==h for p,h in inputs.items());checks['parent_verified']=f.read(ADAM/'FINAL_VERIFY.json')['passed'] and f.read(GRAD/'FINAL_VERIFY.json')['passed'];checks['environment']=str(torch.__version__)==ap['torch'];f.write(out/'PRECHECKS.json',dict(checks=checks,passed=all(checks.values())));assert all(checks.values()),checks
        events=f.tensor_load(OLD/'raw/INPUTS.pt')['events'];grids={n:f.tensor_load(REPO/gp['grids'][str(n)]['path']) for n in [48,384]};predictions=f.read(ADAM/'RESULTS.json');reference=f.read(GRAD/'RESULTS.json')
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No optimizer updates')),patch.object(torch.optim.SGD,'step',side_effect=AssertionError('No optimizer updates')),torch.no_grad():
            for label,lam in zip(NAMES,[0.,1.]):
                cp=f.tensor_load(REPO/ap['checkpoints'][label]['path']);candidate=f.tensor_load(ADAM/'raw'/f'{label}_candidate.pt');base=e.model(cp['model']).requires_grad_(False);probe=e.model(cp['model']).requires_grad_(False);before=e.state_hash(base.state_dict());d=candidate['candidate'];theta=e.vec(base)
                checks[label+'_step']=cp['step']==2000 and candidate['candidate_step']==2001 and candidate['actual_updates']==0;checks[label+'_state_hash']=before==candidate['model_hash'];checks[label+'_order']=candidate['parameters']==[dict(name=n,shape=list(p.shape)) for n,p in base.named_parameters()];assert all(checks.values()),checks
                e.setvec(probe,theta+d);effective=e.vec(probe)-theta;checks[label+'_displacement']=float((effective-d).abs().max())<=1e-14;checks[label+'_buffers']=all(torch.equal(a,dict(probe.named_buffers())[n]) for n,a in base.named_buffers())
                torch.save(dict(state=probe.state_dict(),base_model_hash=before,saved_candidate=d,effective_displacement=effective,alpha=1,diagnostic_only=True,optimizer_steps=0),out/'raw'/f'{label}_probe_state.pt');rows={}
                for n,grid in grids.items():
                    start,raw0=values(base,events,grid,lam);end,raw1=values(probe,events,grid,lam);torch.save(dict(base=raw0,probe=raw1,grid_reference=gp['grids'][str(n)]),out/'raw'/f'{label}_scores_{n}.pt')
                    for point,v in [('base',start),('probe',end)]:checks[label+f'_{n}_{point}_independent']=max(abs(v[k]-v['independent'][k]) for k in ['data','penalty','total','logZ'])<=1e-9
                    checks[label+f'_{n}_archive_base']=abs(start['data']-reference[label]['metrics'][str(n)]['empirical_ppp'])<=1e-9 and abs(start['total']-reference[label]['metrics'][str(n)]['total'])<=1e-9
                    gradients=f.tensor_load(GRAD/'raw'/f'{label}_gradients_{n}.pt');changes={}
                    for kind in ['data','penalty','total']:
                        predicted=predictions[label]['grids'][str(n)][kind]['all']['dot'];actual=end[kind]-start[kind];dot=float(gradients[kind]@d);effective_dot=float(gradients[kind]@effective);checks[label+f'_{n}_{kind}_dot']=abs(dot-predicted)<=1e-10 and abs(effective_dot-predicted)<=1e-9
                        changes[kind]=dict(predicted=predicted,actual=actual,residual=actual-predicted,effective_displacement_prediction=effective_dot,actual_sign=classify(actual),first_order_decrease_but_finite_increase=predicted<-1e-8 and actual>1e-8)
                    checks[label+f'_{n}_decomposition']=abs(changes['total']['actual']-changes['data']['actual']-changes['penalty']['actual'])<=1e-9
                    rows[str(n)]=dict(base=start,probe=end,changes=changes);print(label,n,json.dumps(changes['total']),flush=True)
                checks[label+'_base_unchanged']=e.state_hash(base.state_dict())==before==e.state_hash(cp['model']);checks[label+'_optimizer_unchanged']=all(int(s['step'])==2000 for s in cp['optimizer']['state'].values())
                results[label]=dict(base_model_hash=before,probe_model_hash=e.state_hash(probe.state_dict()),temporary_displacements=1,optimizer_steps=0,displacement_rounding_max=float((effective-d).abs().max()),grids=rows)
        checks['inputs_unchanged']=all(f.sha(REPO/p)==h for p,h in inputs.items());checks['source_unchanged']=all(f.sha(REPO/p)==h for p,h in sources.items());checks['protocol_unchanged']=f.sha(out/'PROTOCOL.json')==prov['protocol_sha256'];changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h];checks['historical_unchanged']=not changed
        f.write(out/'RESULTS.json',results);ver=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],frozen_count=len(frozen),changed=changed,optimizer_steps=0,temporary_displacements=2);f.write(out/'FINAL_VERIFY.json',ver)
        f.write(out/'SUMMARY.json',dict(run_id=out.name,integrity_passed=ver['passed'],optimizer_steps=0,temporary_displacements=2,results={k:{n:rr['changes'] for n,rr in v['grids'].items()} for k,v in results.items()},caution='Isolated finite perturbation on copies, not historical step2001 or additional training. No claim about global or historical cause.'));assert ver['passed'],ver
    except BaseException:
        (out/'logs/error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:f.write(out/'FROZEN_AFTER.json',dict(count=len(frozen),changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h]))
    print('OUT',f.rel(out),flush=True)
if __name__=='__main__':
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run(ROOT/'p1g_finite_step'/f'{stamp}_p1g_candidate_fullstep')
