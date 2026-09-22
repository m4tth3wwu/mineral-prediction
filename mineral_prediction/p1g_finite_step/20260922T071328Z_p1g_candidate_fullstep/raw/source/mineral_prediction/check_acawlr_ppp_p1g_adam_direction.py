"""Read-only Adam candidate displacement from frozen step2000 state; no update."""
import datetime,importlib,inspect,json,math,subprocess,sys,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import acawlr_ppp_p1f_population as f
e=f.e;ROOT,REPO,MEM=f.ROOT,f.REPO,f.MEM
OLD=ROOT/'p1g_empirical/20260922T054836Z_p1g_v1';GRAD=ROOT/'p1g_gradient/20260922T065517Z_p1g_grad48_384';NAMES=['empirical_P0','empirical_Plambda']

def evaluate_direction(g,d):
    normg,normd=float(g.norm()),float(d.norm());dot=float(g@d);bound=1e-12*max(1.,normg*normd)
    return dict(dot=dot,unit_direction_derivative=dot/normd if normd else None,gradient_norm=normg,direction_norm=normd,cosine=dot/(normg*normd) if normg*normd else None,classification='descent' if dot < -bound else 'ascent' if dot>bound else 'numerically_neutral',sign_tolerance=bound,independent_dot=math.fsum(a*b for a,b in zip(g.tolist(),d.tolist())))

def run(out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    for n in ['raw/source','logs']:(out/n).mkdir(parents=True)
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode('utf-8').strip('\0').split('\0');exclude={'docs/project_memory/'+n for n in ['CATCH_UP.md','DECISIONS.md','RUN_INDEX.jsonl']};frozen={p:f.sha(REPO/p) for p in paths if p not in exclude};f.write(out/'FROZEN_BEFORE.json',frozen)
    sources={f.rel(Path(__file__)):f.sha(__file__)}
    for p in f.read(GRAD/'SOURCE_MANIFEST.json'):sources[p]=f.sha(REPO/p)
    for p in sources:
        dest=out/'raw/source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/p).read_bytes())
    f.write(out/'SOURCE_MANIFEST.json',sources);runtime=Path(inspect.getsourcefile(importlib.import_module('torch.optim.adam')));dest=out/'raw/source/runtime/torch_adam.py';dest.parent.mkdir(parents=True);dest.write_bytes(runtime.read_bytes())
    protocol=dict(stage='P1-G frozen endpoint candidate Adam direction',time=f.utc(),run_id=out.name,models=NAMES,archive_step=2000,candidate_step=2001,request='User continues proposed read-only next Adam candidate from archived state and saved48 total gradients.',formula='m_next=beta1*m+(1-beta1)*g48; v_next=beta2*v+(1-beta2)*g48^2; d=-lr*(m_next/(1-beta1^2001))/(sqrt(v_next/(1-beta2^2001))+eps)',comparisons='same candidate d dot data/total/penalty gradients on48 and384, split beta/u/theta; sign describes local derivative only',sign_tolerance='1e-12*max(1,||g||||d||)',verification='tensor reconstruction vs independent scalar formula max abs<=1e-14+1e-12*max|d|; independent scalar dots; exact input/state/history hashes',scope='Zero optimizer steps, no parameter addition, no forward/backward evaluation, no finite-step probing, no new grid. Candidate is not a historical executed update, no prediction of actual finite-step loss or collapse cause.',precision='CPU float64 one thread',dense_caution='384 gradient is a comparison grid; its convergence has not been established.')
    f.write(out/'PROTOCOL.json',protocol)
    gp=f.read(GRAD/'PROVENANCE.json');prov=dict(time=f.utc(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),torch=str(torch.__version__),runtime_adam_path=str(runtime),runtime_adam_sha256=f.sha(runtime),protocol_sha256=f.sha(out/'PROTOCOL.json'),source_manifest_sha256=f.sha(out/'SOURCE_MANIFEST.json'),checkpoints=gp['checkpoints'],gradient_files={k:{str(n):dict(path=f.rel(GRAD/'raw'/f'{k}_gradients_{n}.pt'),sha256=f.sha(GRAD/'raw'/f'{k}_gradients_{n}.pt')) for n in [48,384]} for k in NAMES})
    f.write(out/'PROVENANCE.json',prov);checks={};results={}
    try:
        assert f.read(GRAD/'FINAL_VERIFY.json')['passed'];checks['torch_version']=str(torch.__version__)==gp['torch']
        cps={k:f.tensor_load(REPO/prov['checkpoints'][k]['path']) for k in NAMES};checks['checkpoint_files']=all(f.sha(REPO/v['path'])==v['sha256'] for v in prov['checkpoints'].values())
        f.write(out/'PRECHECKS.json',dict(checks=checks,passed=all(checks.values()),updates=0));assert all(checks.values()),checks
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Forbidden update')),patch.object(torch.optim.SGD,'step',side_effect=AssertionError('Forbidden update')),torch.no_grad():
            for label,cp in cps.items():
                gv={n:f.tensor_load(GRAD/'raw'/f'{label}_gradients_{n}.pt') for n in [48,384]};pg=cp['optimizer']['param_groups'][0];ids=pg['params'];desc=gv[48]['parameters'];g48=gv[48]['total'];g384=gv[384]['total'];before=e.state_hash(cp['model'])
                checks[label+'_state_hash']=before==gv[48]['model_hash']==gv[384]['model_hash'];checks[label+'_order']=desc==gv[384]['parameters'] and len(desc)==len(ids)
                checks[label+'_recipe']=cp['step']==2000 and pg['lr']==.003 and tuple(pg['betas'])==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0 and not pg['amsgrad'] and not pg.get('maximize',False) and not pg.get('capturable',False) and not pg.get('differentiable',False)
                assert all(checks.values()),checks
                moment=[];variance=[];cursor=0;state_steps=[]
                for pid,p in zip(ids,desc):
                    state=cp['optimizer']['state'][pid];shape=tuple(p['shape']);assert tuple(cp['model'][p['name']].shape)==shape==tuple(state['exp_avg'].shape)==tuple(state['exp_avg_sq'].shape)
                    moment.append(state['exp_avg'].flatten().clone());variance.append(state['exp_avg_sq'].flatten().clone());state_steps.append(int(state['step']));cursor+=state['exp_avg'].numel()
                checks[label+'_moment_steps']=set(state_steps)=={2000};checks[label+'_parameter_count']=cursor==len(g48)==675;assert all(checks.values()),checks
                mom=torch.cat(moment);var=torch.cat(variance);b1,b2=pg['betas'];lr,eps=pg['lr'],pg['eps'];t=2001;bc1,bc2=1-b1**t,1-b2**t
                mn=mom.clone().lerp_(g48,1-b1);vn=var.clone().mul_(b2).addcmul_(g48,g48,value=1-b2);den=vn.sqrt()/math.sqrt(bc2)+eps;d=(-lr/bc1)*mn/den
                scalar=[-lr*((b1*a+(1-b1)*g)/bc1)/(math.sqrt((b2*b+(1-b2)*g*g)/bc2)+eps) for a,b,g in zip(mom.tolist(),var.tolist(),g48.tolist())];error=max(abs(x-y) for x,y in zip(d.tolist(),scalar));checks[label+'_scalar_formula']=error<=1e-14+1e-12*float(d.abs().max());checks[label+'_finite']=bool(torch.isfinite(d).all())
                gridrows={}
                for n in [48,384]:
                    gridrows[str(n)]={}
                    for kind in ['total','data','penalty']:
                        gradient=gv[n][kind];row={group:evaluate_direction(gradient[sl],d[sl]) for group,sl in [('all',slice(None)),('beta',slice(0,2)),('u',slice(2,4)),('theta',slice(4,None))]};gridrows[str(n)][kind]=row
                        checks[label+f'_{kind}_{n}_dot']=all(abs(z['dot']-z['independent_dot'])<=1e-12*max(1.,abs(z['dot'])) for z in row.values())
                        checks[label+f'_{kind}_{n}_blocks']=abs(sum(row[k]['dot'] for k in ['beta','u','theta'])-row['all']['dot'])<=1e-12*max(1.,abs(row['all']['dot']))
                    checks[label+f'_decomposition{n}']=abs(gridrows[str(n)]['total']['all']['dot']-gridrows[str(n)]['data']['all']['dot']-gridrows[str(n)]['penalty']['all']['dot'])<=1e-12*max(1.,abs(gridrows[str(n)]['total']['all']['dot']))
                checks[label+'_model_unchanged']=e.state_hash(cp['model'])==before
                checks[label+'_moments_unchanged']=torch.equal(torch.cat([cp['optimizer']['state'][pid]['exp_avg'].flatten() for pid in ids]),mom) and torch.equal(torch.cat([cp['optimizer']['state'][pid]['exp_avg_sq'].flatten() for pid in ids]),var) and all(int(cp['optimizer']['state'][pid]['step'])==2000 for pid in ids)
                torch.save(dict(candidate=d,moment_old=mom,variance_old=var,moment_candidate=mn,variance_candidate=vn,denominator=den,g48_total=g48,g384_total=g384,parameters=desc,model_hash=before,candidate_step=2001,actual_updates=0),out/'raw'/f'{label}_candidate.pt')
                results[label]=dict(candidate_norm=float(d.norm()),scalar_formula_max_error=error,source_step=2000,candidate_step=2001,actual_updates=0,grids=gridrows);print(label,json.dumps({n:gridrows[n]['total']['all'] for n in ['48','384']}),flush=True)
        checks['runtime_source_unchanged']=f.sha(runtime)==prov['runtime_adam_sha256'];checks['source_unchanged']=all(f.sha(REPO/p)==h for p,h in sources.items());checks['protocol_unchanged']=f.sha(out/'PROTOCOL.json')==prov['protocol_sha256'];changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h];checks['historical_unchanged']=not changed
        f.write(out/'RESULTS.json',results);ver=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],frozen_count=len(frozen),changed=changed,actual_updates=0);f.write(out/'FINAL_VERIFY.json',ver)
        f.write(out/'SUMMARY.json',dict(run_id=out.name,actual_updates=0,integrity_passed=ver['passed'],results={k:dict(candidate_norm=v['candidate_norm'],total={n:v['grids'][n]['total']['all'] for n in ['48','384']},data={n:v['grids'][n]['data']['all'] for n in ['48','384']}) for k,v in results.items()},caution='Local candidate directional derivative, not executed update or finite-step prediction. No historical-cause or converged-gradient claim.'));assert ver['passed'],ver
    except BaseException:
        (out/'logs/error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:f.write(out/'FROZEN_AFTER.json',dict(count=len(frozen),changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h]))
    print('OUT',f.rel(out),flush=True)
if __name__=='__main__':
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run(ROOT/'p1g_adam_direction'/f'{stamp}_p1g_adam_candidate')
