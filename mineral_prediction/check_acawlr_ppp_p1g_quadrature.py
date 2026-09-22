"""Fixed P1-G endpoint quadrature audit, no training and no adaptive refinement."""
import datetime,json,math,subprocess,sys,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import acawlr_ppp_p1f_population as f
e=f.e;ROOT=f.ROOT;REPO=f.REPO;MEM=f.MEM
OLD=ROOT/'p1g_empirical/20260922T054836Z_p1g_v1';POP=f.ROOT/'p1f_population/20260917T121240Z_p1f_v1'
NAMES=['empirical_P0','empirical_Plambda']

def independent(s,truth,w):
    weights=w.tolist();gs=s['g'].tolist();gt=truth['g'].tolist()
    def z(v):
        top=max(v);return top+math.log(math.fsum(a*math.exp(b-top) for a,b in zip(weights,v)))
    zs,zt=z(gs),z(gt);pt=[a*math.exp(b-zt) for a,b in zip(weights,gt)];ps=[a*math.exp(b-zs) for a,b in zip(weights,gs)]
    return dict(logZ=zs,KL=math.fsum(p*(t-g) for p,t,g in zip(pt,gt,gs))+zs-zt,TV=math.fsum(abs(a-b) for a,b in zip(pt,ps))/2)

def build_grid(teacher,n):
    xy,w=e.quadrature(n);s=e.surface(teacher,xy);t=e.target(s['g'],w);a=w/w.sum();q=(xy/e.SCALE@teacher.u)*s['h']
    t.update(xy=xy,teacher_beta=teacher.beta.detach(),teacher_theta=e.vec(teacher)[4:],teacher_q_RMS=float((a*q.square()).sum().sqrt()),teacher_field_RMS=float((a[:,None]*(s['h'][:,None]*teacher.u).square()).sum().sqrt()),teacher_canonical=e.canonical(teacher,s['h'],w))
    return dict(xy=xy,w=w,truth=s,target=t)

def run(out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    for n in ['raw/source','logs']:(out/n).mkdir(parents=True)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().strip('\0').split('\0');exclude={'docs/project_memory/'+n for n in ['CATCH_UP.md','DECISIONS.md','RUN_INDEX.jsonl']}
    frozen={p:f.sha(REPO/p) for p in tracked if p not in exclude};f.write(out/'FROZEN_BEFORE.json',frozen)
    sourcepaths=[Path(__file__),ROOT/'acawlr_ppp_p1f_population.py',ROOT/'acawlr_ppp_p1e_identifiability.py',ROOT/'acawlr_ppp_bridge.py',ROOT/'acawlr_ppp_synthetic.py',ROOT/'acawlr_ppp_teacher_student.py',REPO/'scripts/ACAWLR_improved.py'];sources={f.rel(p):f.sha(p) for p in sourcepaths};f.write(out/'SOURCE_MANIFEST.json',sources)
    for p in sourcepaths:
        dest=out/'raw/source'/f.rel(p);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    prot=dict(stage='P1-G fixed endpoint numerical follow-up',time=f.utc(),run_id=out.name,parent=f.rel(OLD),request='User: 继续下一步; accepted proposal: fixed two step2000 models, compare192/384 without training or parameter changes.',grids=[192,384],models=NAMES,updates=0,dtype='float64',threads=1,quadrature='unchanged nonuniform midpoint cells, true area weights',thresholds=dict(logZ_abs=1e-4,KL_abs=1e-6,KL_relative=.01,TV_abs=1e-3),policy='No higher grid, optimizer, parameter modification or adaptive refinement. Preserve old failed gates. New pairwise pass is not exact continuous integration proof or stationarity.',verification='192 surfaces bit equal archived; metrics reproduce old within1e-12; independent standard-library logZ/KL/TV agreement1e-10; all model and historical hashes unchanged.')
    f.write(out/'PROTOCOL.json',prot)
    prov=dict(time=f.utc(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),torch=str(torch.__version__),python=sys.version,protocol_sha256=f.sha(out/'PROTOCOL.json'),source_manifest_sha256=f.sha(out/'SOURCE_MANIFEST.json'),input_file_sha256=f.sha(OLD/'raw/INPUTS.pt'),checkpoint_files={k:dict(path=f.rel(OLD/'raw/states'/f'{k}_2000.pt'),sha256=f.sha(OLD/'raw/states'/f'{k}_2000.pt')) for k in NAMES})
    f.write(out/'PROVENANCE.json',prov);checks={};results={}
    try:
        inp=f.tensor_load(OLD/'raw/INPUTS.pt');oldprov=f.read(OLD/'PROVENANCE.json');checks['input_file']=prov['input_file_sha256']==oldprov['input_file_sha256'];checks['teacher']=e.state_hash(inp['teacher'])==e.TEACHER_HASH==oldprov['teacher_sha256'];checks['events']=e.state_hash({'events':inp['events']})==oldprov['data_sha256'];checks['environment']=str(torch.__version__)==oldprov['torch']
        cps={k:f.tensor_load(OLD/'raw/states'/f'{k}_2000.pt') for k in NAMES};checks['fixed_step2000']=all(cp['step']==2000 and all(int(v['step'])==2000 for v in cp['optimizer']['state'].values()) for cp in cps.values())
        f.write(out/'PRECHECKS.json',dict(checks=checks,passed=all(checks.values()),updates=0));assert all(checks.values()),checks
        teacher=e.model(inp['teacher']).requires_grad_(False);models={k:e.model(cp['model']).requires_grad_(False) for k,cp in cps.items()};hashes={k:e.state_hash(m.state_dict()) for k,m in models.items()};prov['model_state_hashes']=hashes;f.write(out/'PROVENANCE.json',prov)
        old=f.read(OLD/'EMPIRICAL_COMPARISON.json');delta=f.read(POP/'CONTROLS.json')['192']['global']['KL'];grids={192:f.tensor_load(POP/'raw/GRID_192.pt')}
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No optimizer updates authorized')),patch.object(torch.optim.SGD,'step',side_effect=AssertionError('No optimizer updates authorized')),torch.no_grad():
            grids[384]=build_grid(teacher,384);torch.save(grids[384],out/'raw/GRID_384.pt')
            for k,m in models.items():
                rows={};ver={};es=float(m.g(inp['events']).sum())
                for n,z in grids.items():
                    surface=e.surface(m,z['xy']);v=f.evaluate_arrays(m,surface,z['target'],z['truth'],z['w']);v['labels']=f.labels(v,z['target'],delta);v['empirical_ppp']=256*v['logZ']-es;rows[str(n)]=v
                    torch.save(surface,out/'raw'/f'{k}_surface_{n}.pt');ind=independent(surface,z['truth'],z['w']);errors={x:abs(ind[x]-v[x]) for x in ind};checks[k+f'_math{n}']=max(errors.values())<1e-10;ver[str(n)]=dict(independent=ind,errors=errors)
                    if n==192:
                        saved=f.tensor_load(OLD/'raw'/f'{k}_surface_192.pt');checks[k+'_surface192_exact']=all(torch.equal(surface[x],saved[x]) for x in surface);checks[k+'_metrics192']=all(abs(v[x]-old[k]['final']['192'][x])<1e-12 for x in ['KL','TV','logZ','q_ratio','field_ratio','b_RMSE'])
                checks[k+'_state_unchanged']=e.state_hash(m.state_dict())==hashes[k];results[k]=dict(metrics=rows,fidelity=e.fidelity(rows['192'],rows['384']),verification=ver,empirical_objective_shift=rows['384']['empirical_ppp']-rows['192']['empirical_ppp'],stationarity='not reevaluated; frozen P1-G status remains false')
                print(k,json.dumps(dict(fidelity=results[k]['fidelity'],KL384=rows['384']['KL'],q384=rows['384']['q_ratio'],field384=rows['384']['field_ratio'])),flush=True)
        checks['teacher_unchanged']=e.state_hash(teacher.state_dict())==e.TEACHER_HASH;checks['source_unchanged']=all(f.sha(REPO/p)==h for p,h in sources.items());checks['protocol_unchanged']=f.sha(out/'PROTOCOL.json')==prov['protocol_sha256']
        changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h];checks['historical_unchanged']=not changed
        teacher_gap=abs(float(grids[384]['target']['logz']-grids[192]['target']['logz']));checks['teacher_logZ']=teacher_gap<=1e-4
        f.write(out/'RESULTS.json',results);f.write(out/'GATES.json',dict(endpoint_fidelity={k:v['fidelity'] for k,v in results.items()},teacher_logZ_abs=teacher_gap,teacher_logZ_passed=teacher_gap<=1e-4,all_endpoints_pass=all(v['fidelity']['passed'] for v in results.values())))
        verification=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],frozen_count=len(frozen),changed=changed,updates=0);f.write(out/'FINAL_VERIFY.json',verification)
        effects={metric:{str(n):results['empirical_Plambda']['metrics'][str(n)][metric]-results['empirical_P0']['metrics'][str(n)][metric] for n in [192,384]} for metric in ['KL','TV','q_ratio','field_ratio','b_RMSE','centered_g_area_RMSE']};f.write(out/'EFFECT_STABILITY.json',effects)
        summary=dict(run_id=out.name,parent=f.rel(OLD),updates=0,execution='COMPLETED',integrity=verification['passed'],all_endpoints_pass=all(v['fidelity']['passed'] for v in results.values()),results={k:dict(fidelity=v['fidelity'],metrics384=v['metrics']['384']) for k,v in results.items()},effects=effects)
        f.write(out/'SUMMARY.json',summary);assert verification['passed'],verification
    except BaseException:
        (out/'logs/error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        f.write(out/'FROZEN_AFTER.json',dict(count=len(frozen),changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h]))
    print('OUT',f.rel(out),flush=True)
if __name__=='__main__':
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run(ROOT/'p1g_quadrature'/f'{stamp}_p1g_q192_384')
