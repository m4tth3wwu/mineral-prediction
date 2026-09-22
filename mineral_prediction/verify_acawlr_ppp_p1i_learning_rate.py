"""Independent P1-I record verification, no optimizer updates or extra training."""
import json,math,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import acawlr_ppp_p1i_learning_rate as h
f,e=h.f,h.e;REPO,ROOT=f.REPO,f.ROOT

def run(out):
    torch.set_num_threads(1);checks={};detail={};prov=h.read(out/'PROVENANCE.json');sources=h.read(out/'SOURCE_MANIFEST.json');checks['source_frozen']=all(f.sha(REPO/p)==v for p,v in sources.items());checks['protocol_frozen']=f.sha(out/'PROTOCOL.json')==prov['protocol_sha256'];checks['grids_frozen']=all(f.sha(REPO/p)==v for p,v in prov['grids'].items());checks['input_file']=f.sha(out/'raw/INPUTS.pt')==prov['input_file_sha256'];checks['pretraining_pass']=h.read(out/'GATES.json')['pretraining']['passed'];checks['training_count']=h.read(out/'TRAINING_END.json')['updates']==4000;checks['evaluation_pass']=h.read(out/'EVALUATION_VERIFY.json')['passed']
    inp=f.tensor_load(out/'raw/INPUTS.pt');z=h.grid(96);checks['input_hash']=e.state_hash({'events':inp['events']})==prov['data_sha256'] and e.state_hash(inp['teacher'])==e.TEACHER_HASH and all(e.state_hash(v)==e.INIT_HASH for v in inp['starts'].values())
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No verification updates')):
        for label,lam in zip(h.NAMES,[0.,1.]):
            rows=[json.loads(x) for x in (out/'raw'/f'{label}_trajectory.jsonl').read_text(encoding='utf-8').splitlines()];obs=h.read(out/'raw'/f'{label}_observations.json');checks[label+'_rows']=len(rows)==2001 and [v['step'] for v in rows]==list(range(2001));checks[label+'_monitors']=[v['step'] for v in obs]==list(range(0,2001,25));checks[label+'_grid_labels']=all(v['training_grid']==96 and v['diagnostic_grid']==48 for v in rows)
            checks[label+'_accounting']=all(abs(v['empirical_ppp']+v['penalty_total']-v['total_objective'])<=1e-9 and abs(sum(v['penalty_effective'].values())-v['penalty_total'])<=1e-9 and all(abs(lam*v['penalty_reference'][k]-v['penalty_effective'][k])<=1e-12 for k in ['beta','u','theta']) for v in rows)
            checks[label+'_linear_accounting']=rows[0]['linear'] is None and all(abs(v['linear']['actual']-(v['total_objective']-p['total_objective']))<=1e-12 and abs(v['linear']['remainder']-(v['linear']['actual']-v['linear']['prediction']))<=1e-12 and v['linear']['reversal']==(v['linear']['prediction'] < -1e-8 and v['linear']['actual'] >1e-8) for p,v in zip(rows,rows[1:]))
            checks[label+'_norm_decomposition']=all(abs(sum(v['gradient'][k]**2 for k in ['beta','u','theta'])-v['gradient']['all']**2)<=1e-8*max(1.,v['gradient']['all']**2) and abs(sum(v['actual_update'][k]**2 for k in ['beta','u','theta'])-v['actual_update']['all']**2)<=1e-12 for v in rows)
            checks[label+'_flags']=all(v['flags']==h.flags(rows[:i],v) for i,v in enumerate(rows));checks[label+'_monitor_alignment']=all(abs(v['total_objective']-rows[v['step']]['total_objective'])<=1e-12 and abs(v['empirical_ppp_96']-rows[v['step']]['empirical_ppp'])<=1e-9 for v in obs)
            paths=list((out/'raw/states').glob(label+'_*.pt'));states={int(p.stem.rsplit('_',1)[1]):p for p in paths};required=set(range(0,2001,25))
            for row in rows:
                if any(row['flags'].values()):required.update(range(max(0,row['step']-2),min(2000,row['step']+2)+1))
            checks[label+'_checkpoint_coverage']=required==set(states);cp_errors=[];previous=None;pair_count=0
            for step,path in sorted(states.items()):
                cp=f.tensor_load(path);m=e.model(cp['model']);vec=e.vec(m);pg=cp['optimizer']['param_groups'][0]
                if pg['lr']!=.001 or tuple(pg['betas'])!=(.9,.999) or pg['eps']!=1e-8 or pg['weight_decay']!=0 or pg['amsgrad']:cp_errors.append([step,'optimizer_recipe'])
                if cp['step']!=step or cp['lambda_multiplier']!=lam or cp['initial_sha256']!=e.INIT_HASH:cp_errors.append([step,'metadata'])
                if step==0:
                    if cp['optimizer']['state'] or e.state_hash(cp['model'])!=e.INIT_HASH:cp_errors.append([step,'initial'])
                elif any(int(v['step'])!=step for v in cp['optimizer']['state'].values()):cp_errors.append([step,'optimizer_step'])
                if abs(float(vec.norm())-rows[step]['parameter_norm'])>1e-10:cp_errors.append([step,'norm'])
                if previous is not None and previous[0]==step-1:
                    pair_count+=1;diff=vec-previous[1]
                    if any(abs(float(diff[sl].norm())-rows[step]['actual_update'][key])>1e-12 for key,sl in [('all',slice(None)),('beta',slice(0,2)),('u',slice(2,4)),('theta',slice(4,None))]):cp_errors.append([step,'actual_update'])
                previous=(step,vec)
            checks[label+'_checkpoint_integrity']=not cp_errors;verification_points=[]
            for step in [0,1000,1975]:
                cp=f.tensor_load(states[step]);m=e.model(cp['model']);loss=h.empirical(m,inp['events'],z['xy'],z['w'])+lam*e.PENALTY(m);gr=e.flat_grad(loss,m);pg=cp['optimizer']['param_groups'][0];ids=pg['params'];ss=cp['optimizer']['state'];b1,b2=pg['betas'];t=step+1
                mom=torch.cat([ss[i]['exp_avg'].flatten() for i in ids]) if step else torch.zeros_like(gr);var=torch.cat([ss[i]['exp_avg_sq'].flatten() for i in ids]) if step else torch.zeros_like(gr)
                candidate=torch.tensor([-pg['lr']*((b1*a+(1-b1)*g)/(1-b1**t))/(math.sqrt((b2*b+(1-b2)*g*g)/(1-b2**t))+pg['eps']) for a,b,g in zip(mom.tolist(),var.tolist(),gr.tolist())],dtype=torch.float64)
                predicted=math.fsum(a*b for a,b in zip(gr.tolist(),candidate.tolist()));nr=rows[step+1];errors=dict(objective=abs(float(loss.detach())-rows[step]['total_objective']),gradient=abs(float(gr.norm())-rows[step]['gradient']['all']),prediction=abs(predicted-nr['linear']['prediction']),update=max(abs(float(candidate[sl].norm())-nr['actual_update'][key]) for key,sl in [('all',slice(None)),('beta',slice(0,2)),('u',slice(2,4)),('theta',slice(4,None))]));checks[f'{label}_alignment_{step}']=errors['objective']<=1e-9 and errors['gradient']<=1e-8 and errors['prediction']<=1e-8 and errors['update']<=1e-10;verification_points.append(dict(step=step,errors=errors))
            detail[label]=dict(checkpoints=len(paths),consecutive_saved_pairs=pair_count,checkpoint_errors=cp_errors,alignment_points=verification_points)
    report=h.read(out/'SUMMARY.json')['runs'];contrast=h.read(out/'COMPARISON.json');endpoints=h.read(out/'ENDPOINTS.json')
    for lr,folder in [(.003,h.REFERENCE),(.001,out)]:
        for label in h.NAMES:
            key=f'lr{lr:.3f}_{label}';rr=[json.loads(x) for x in (folder/'raw'/f'{label}_trajectory.jsonl').read_text(encoding='utf-8').splitlines()][1:];cnt=dict(reversals=sum(x['linear']['prediction'] < -1e-8 and x['linear']['actual'] >1e-8 for x in rr),positive_predictions=sum(x['linear']['prediction']>1e-8 for x in rr),actual_increases=sum(x['linear']['actual']>1e-8 for x in rr),positive_prediction_and_increase=sum(x['linear']['prediction']>1e-8 and x['linear']['actual']>1e-8 for x in rr));checks[key+'_reported_rates']=cnt==report[key]['linear_counts'] and all(report[key]['linear_rates'][k]==v/2000 for k,v in cnt.items());checks[key+'_reported_endpoint']=report[key]['metrics768']==endpoints[key]['metrics']['768'] and report[key]['learning_rate']==lr and report[key]['training_grid']==96
    for n in ['384','768']:
        for label in h.NAMES:
            checks[label+'_contrast'+n]=all(abs(v['difference']-(endpoints['lr0.001_'+label]['metrics'][n][k]-endpoints['lr0.003_'+label]['metrics'][n][k]))<=1e-12 for k,v in contrast[n]['lr_effect'][label].items())
    frozen=h.frozen_check();checks['historical_unchanged']=frozen['passed'];ver=dict(time=h.utc(),passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],detail=detail,frozen=frozen,optimizer_steps=0);h.write(out/'FINAL_VERIFY.json',ver);h.record(out,'independent_verification','PASSED' if ver['passed'] else 'FAILED',dict(checks=len(checks),failed=ver['failed'],frozen=frozen));print(json.dumps(ver),flush=True);assert ver['passed'],ver
if __name__=='__main__':
    out=REPO/h.session()['output_dir']
    try:run(out)
    except BaseException:
        (out/'logs/verification_error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
