"""Single frozen P1-G lambda0 endpoint:384/768 quadrature, zero updates."""
import datetime,json,subprocess,sys,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import check_acawlr_ppp_p1g_quadrature as q
f,e=q.f,q.e;ROOT,REPO,MEM=q.ROOT,q.REPO,q.MEM
PREV=ROOT/'p1g_quadrature/20260922T062242Z_p1g_q192_384';LABEL='empirical_P0'

def run(out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    for n in ['raw/source','logs']:(out/n).mkdir(parents=True)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode('utf-8').strip('\0').split('\0');excluded={'docs/project_memory/'+n for n in ['CATCH_UP.md','DECISIONS.md','RUN_INDEX.jsonl']};frozen={p:f.sha(REPO/p) for p in tracked if p not in excluded};f.write(out/'FROZEN_BEFORE.json',frozen)
    sources={f.rel(Path(__file__)):f.sha(__file__)}
    for p in [*f.read(PREV/'SOURCE_MANIFEST.json'),f.rel(ROOT/'check_acawlr_ppp_p1g_quadrature.py')]:sources[p]=f.sha(REPO/p)
    for p in sources:
        dest=out/'raw/source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/p).read_bytes())
    f.write(out/'SOURCE_MANIFEST.json',sources)
    protocol=dict(stage='P1-G lambda0 fixed-endpoint384/768 follow-up',run_id=out.name,time=f.utc(),request='User: 继续; accepts the proposed lambda0-only384/768 check. No original-lambda reevaluation.',checkpoint=f.rel(q.OLD/'raw/states/empirical_P0_2000.pt'),grids=[384,768],updates=0,models=[LABEL],dtype='float64',threads=1,thresholds=dict(logZ_abs=1e-4,KL_abs=1e-6,KL_relative=.01,TV_abs=1e-3),policy='Same quadrature rule and fixed step2000 model; no parameter changes, training, adaptive refinement or new seeds. Preserve all historical failed gates. Pairwise grid agreement does not prove exact integration or optimizer convergence.',verification='384 surface exact against previous; independent math.fsum/log/exp verification at both resolutions; frozen source/state/history hashes.')
    f.write(out/'PROTOCOL.json',protocol);checkpoint=q.OLD/'raw/states/empirical_P0_2000.pt'
    prov=dict(time=f.utc(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=subprocess.list2cmdline([sys.executable,'-B',*sys.argv]),torch=str(torch.__version__),python=sys.version,checkpoint_sha256=f.sha(checkpoint),input_sha256=f.sha(q.OLD/'raw/INPUTS.pt'),protocol_sha256=f.sha(out/'PROTOCOL.json'),source_manifest_sha256=f.sha(out/'SOURCE_MANIFEST.json'))
    f.write(out/'PROVENANCE.json',prov);checks={}
    try:
        inp=f.tensor_load(q.OLD/'raw/INPUTS.pt');cp=f.tensor_load(checkpoint);oldprov=f.read(q.OLD/'PROVENANCE.json');pv=f.read(PREV/'PROVENANCE.json')
        checks.update(input_exact=prov['input_sha256']==oldprov['input_file_sha256'],checkpoint_exact=prov['checkpoint_sha256']==pv['checkpoint_files'][LABEL]['sha256'],step2000=cp['step']==2000 and all(int(v['step'])==2000 for v in cp['optimizer']['state'].values()),teacher=e.state_hash(inp['teacher'])==e.TEACHER_HASH,events=e.state_hash({'events':inp['events']})==oldprov['data_sha256'],environment=str(torch.__version__)==oldprov['torch'])
        f.write(out/'PRECHECKS.json',dict(checks=checks,passed=all(checks.values()),updates=0));assert all(checks.values()),checks
        m=e.model(cp['model']).requires_grad_(False);teacher=e.model(inp['teacher']).requires_grad_(False);statehash=e.state_hash(m.state_dict());assert statehash==pv['model_state_hashes'][LABEL];prov['model_state_hash']=statehash;f.write(out/'PROVENANCE.json',prov)
        grids={384:f.tensor_load(PREV/'raw/GRID_384.pt')};rows={};independent={};delta=f.read(q.POP/'CONTROLS.json')['192']['global']['KL']
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Updates forbidden')),patch.object(torch.optim.SGD,'step',side_effect=AssertionError('Updates forbidden')),torch.no_grad():
            grids[768]=q.build_grid(teacher,768);z=grids[768]
            torch.save(dict(xy=z['xy'],w=z['w']),out/'raw/GRID_768_coordinates.pt');torch.save(z['truth'],out/'raw/GRID_768_truth.pt');torch.save({k:v for k,v in z['target'].items() if k!='xy'},out/'raw/GRID_768_target.pt')
            es=float(m.g(inp['events']).sum())
            for n,z in grids.items():
                surf=e.surface(m,z['xy']);v=f.evaluate_arrays(m,surf,z['target'],z['truth'],z['w']);v['labels']=f.labels(v,z['target'],delta);v['empirical_ppp']=256*v['logZ']-es;rows[str(n)]=v
                torch.save(surf,out/'raw'/f'{LABEL}_surface_{n}.pt');ind=q.independent(surf,z['truth'],z['w']);errors={k:abs(ind[k]-v[k]) for k in ind};checks[f'independent_math{n}']=max(errors.values())<1e-10;independent[str(n)]=dict(values=ind,errors=errors)
                if n==384:
                    saved=f.tensor_load(PREV/'raw/empirical_P0_surface_384.pt');checks['surface384_exact']=all(torch.equal(surf[k],saved[k]) for k in surf);old=f.read(PREV/'RESULTS.json')[LABEL]['metrics']['384'];checks['metrics384_exact']=all(abs(v[k]-old[k])<1e-12 for k in ['logZ','KL','TV','q_ratio','field_ratio','b_RMSE'])
                print(json.dumps(dict(grid=n,KL=v['KL'],q_ratio=v['q_ratio'],field_ratio=v['field_ratio'])),flush=True)
        fidelity=e.fidelity(rows['384'],rows['768']);teacher_gap=abs(float(grids[768]['target']['logz']-grids[384]['target']['logz']));checks['teacher_logZ']=teacher_gap<=1e-4;checks['state_unchanged']=statehash==e.state_hash(m.state_dict());checks['teacher_unchanged']=e.state_hash(teacher.state_dict())==e.TEACHER_HASH;checks['source_unchanged']=all(f.sha(REPO/p)==h for p,h in sources.items());checks['protocol_unchanged']=f.sha(out/'PROTOCOL.json')==prov['protocol_sha256'];changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h];checks['historical_unchanged']=not changed
        f.write(out/'RESULTS.json',dict(model=LABEL,metrics=rows,fidelity=fidelity,independent=independent,empirical_objective_shift=rows['768']['empirical_ppp']-rows['384']['empirical_ppp']))
        f.write(out/'GATES.json',dict(endpoint_fidelity=fidelity,teacher_logZ_abs=teacher_gap,teacher_pass=checks['teacher_logZ']))
        verify=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],frozen_count=len(frozen),changed=changed,updates=0);f.write(out/'FINAL_VERIFY.json',verify)
        f.write(out/'SUMMARY.json',dict(run_id=out.name,execution='COMPLETED',updates=0,integrity_passed=verify['passed'],fidelity=fidelity,metrics768=rows['768'],original_lambda='Not reevaluated; previous192/384 pair passed',stationarity='Unchanged from frozen P1-G; not passed'))
        assert verify['passed'],verify;print(json.dumps(dict(fidelity=fidelity,verified=verify['passed'],frozen_files=len(frozen))),flush=True)
    except BaseException:
        (out/'logs/error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:f.write(out/'FROZEN_AFTER.json',dict(count=len(frozen),changed=[p for p,h in frozen.items() if f.sha(REPO/p)!=h]))
    print('OUT',f.rel(out),flush=True)
if __name__=='__main__':
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run(ROOT/'p1g_quadrature'/f'{stamp}_p1g_q384_768')
