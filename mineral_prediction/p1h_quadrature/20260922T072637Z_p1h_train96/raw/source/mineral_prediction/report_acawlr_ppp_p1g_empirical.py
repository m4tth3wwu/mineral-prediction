"""Read-only P1-G factorial contrasts, predeclared dynamics and plots. No torch/training."""
import json,math,statistics,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
POP=ROOT/'p1f_population/20260917T121240Z_p1f_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def flags(h,i):
    if i<=25:return dict(objective_spike=False,gradient_spike=False,transition_candidate=False,collapse_transition=False)
    r,p=h[i],h[i-1];past=sorted(v['gradient']['all'] for v in h[i-25:i]);med=past[len(past)//2];gr=r['gradient']['all'];col=r['labels']['collapse']!=p['labels']['collapse']
    return dict(objective_spike=abs(r['total_objective']-p['total_objective'])>1,gradient_spike=gr/256>10 or (gr>10*max(med,1e-30) and gr-med>1),transition_candidate=abs(r['q_ratio']-p['q_ratio'])>.1 or abs(r['field_ratio']-p['field_ratio'])>.1 or col,collapse_transition=col)
def dynamics(h):
    fl=[flags(h,i) for i in range(len(h))];out={k:[r['step'] for r,ff in zip(h,fl) if ff[k]] for k in fl[0]}
    for field,key in [('gradient_max','gradient'),('update_max','actual_update')]:
        out[field]=max(r[key]['all'] for r in h);out[field+'_after300']=max(r[key]['all'] for r in h if r['step']>300)
    out['gradient_tail_median']=statistics.median(r['gradient']['all'] for r in h[-200:]);out['total_tail_range']=max(r['total_objective'] for r in h[-200:])-min(r['total_objective'] for r in h[-200:])
    rel=[]
    for i,r in enumerate(h):
        p=h[max(0,i-1)]['penalty_reference'];norm=math.sqrt(p['beta']/.01+p['u']/.1+p['theta']/.0005);rel.append(r['actual_update']['all']/max(norm,1e-30))
    out['relative_update_max']=max(rel);out['relative_update_max_after300']=max(rel[301:]);out['collapse_indicator']=int(any(r['labels']['collapse'] for r in h));out['nan_inf']=False
    for k in fl[0]:out[k+'_count']=len(out[k])
    return out

def conflict(h,obs):
    def interval(a,b,ma,mb):
        d=dict(total=b['total_objective']-a['total_objective'],empirical=b['empirical_ppp']-a['empirical_ppp'],penalty=b['penalty_total']-a['penalty_total'],KL=mb['KL']-ma['KL'],g_RMSE=mb['centered_g_area_RMSE']-ma['centered_g_area_RMSE'],q_ratio=mb['q_ratio']-ma['q_ratio'],field_ratio=mb['field_ratio']-ma['field_ratio'])
        yes=d['total']<-1e-8 and d['empirical']>1e-8 and d['KL']>1e-8 and d['g_RMSE']>1e-8
        over=d['total']<-1e-8 and d['empirical']<-1e-8 and d['KL']>1e-8 and d['g_RMSE']>1e-8
        return dict(start=a['step'],end=b['step'],delta=d,conflict=yes,overfit_pattern=over)
    dense=[interval(a,b,a,b) for a,b in zip(h,h[1:])];mon=[interval(a,b,a['96'],b['96']) for a,b in zip(obs,obs[1:])]
    streak=0;longest=0
    for v in mon:streak=streak+1 if v['conflict'] else 0;longest=max(longest,streak)
    cp=[0,100,300,600,1000,1500,2000];coarse=[]
    for a,b in zip(cp,cp[1:]):
        seq=[v for v in obs if a<=v['step']<=b];v=interval(seq[0],seq[-1],seq[0]['96'],seq[-1]['96']);v['strictly_decreasing_monitor_total']=all(y['total_objective']<x['total_objective']-1e-8 for x,y in zip(seq,seq[1:]));coarse.append(v)
    return dict(step48_conflicts=[v for v in dense if v['conflict']],monitor96_intervals=mon,monitor96_conflicts=[v for v in mon if v['conflict']],longest_monitor_conflict_streak=longest,sustained_conflict=longest>=3,coarse_fixed_intervals=coarse,coarse_sustained_conflict=any(v['conflict'] and v['strictly_decreasing_monitor_total'] for v in coarse),overfit_monitor_intervals=[v for v in mon if v['overfit_pattern']],scope='Monitor96 intervals do not assert monotonicity between sampled observations; step48 separately available.')

def report(out):
    assert read(out/'TRAINING_END.json')['updates']==4000
    emp=read(out/'EMPIRICAL_COMPARISON.json');pop=read(POP/'POPULATION_COMPARISON.json');allruns={k:pop[k] for k in ['standard_P0','standard_Plambda']};allruns.update(emp)
    hist={k:rows((POP if k.startswith('standard') else out)/'raw'/f'{k}_trajectory.jsonl') for k in allruns}
    obs={k:read((POP if k.startswith('standard') else out)/'raw'/f'{k}_observations.json') for k in allruns}
    dyn={k:dynamics(v) for k,v in hist.items()};conf={k:conflict(hist[k],obs[k]) for k in emp}
    write(out/'TRAJECTORY_DYNAMICS.json',dict(runs=dyn,objective_conflict=conf))
    metrics=['KL','TV','centered_g_area_RMSE','centered_g_teacher_RMSE','q_ratio','field_ratio','b_RMSE','beta_error','u_norm','theta_norm']
    interaction={}
    for k in metrics:
        vals={n:r['final']['192'][k] for n,r in allruns.items()};a=vals['empirical_Plambda']-vals['empirical_P0'];b=vals['standard_Plambda']-vals['standard_P0'];interaction[k]=dict(values=vals,empirical_effect=a,population_effect=b,interaction=a-b)
    for k in ['gradient_max','gradient_max_after300','gradient_tail_median','update_max','update_max_after300','relative_update_max','relative_update_max_after300','total_tail_range','objective_spike_count','gradient_spike_count','transition_candidate_count','collapse_indicator']:
        vals={n:r[k] for n,r in dyn.items()};a=vals['empirical_Plambda']-vals['empirical_P0'];b=vals['standard_Plambda']-vals['standard_P0'];interaction[k]=dict(values=vals,empirical_effect=a,population_effect=b,interaction=a-b)
    write(out/'OBJECTIVE_REGULARIZATION_INTERACTION.json',dict(definition='(empirical_lambda-empirical_0)-(population_lambda-population_0)',metrics=interaction,note='Descriptive matched contrast, one fixed dataset and init. No statistical significance or optimized-minimum claim.',endpoint_fidelity={k:v['fidelity'] for k,v in allruns.items()}))
    summary=dict(run_id=out.name,trajectories=2,updates=4000,all_endpoint_fidelity=all(v['fidelity']['passed'] for v in allruns.values()),runs={},conflict={})
    for k,v in allruns.items():
        m=v['final']['192'];summary['runs'][k]={x:m[x] for x in metrics};summary['runs'][k].update(collapse=m['labels']['collapse'],collapse_ever=bool(dyn[k]['collapse_indicator']),joint_functional_failure=m['labels']['collapse'] and not m['labels']['density_success'] and not m['labels']['predictor_success'],stationarity=v['stationarity'],fidelity=v['fidelity'],density_success=v['density_success'],predictor_success=v['predictor_success'],category=m['labels']['category'])
    for k,v in conf.items():summary['conflict'][k]={n:v[n] for n in ['sustained_conflict','coarse_sustained_conflict','longest_monitor_conflict_streak']};summary['conflict'][k].update(step48_count=len(v['step48_conflicts']),monitor96_count=len(v['monitor96_conflicts']),overfit_monitor_count=len(v['overfit_monitor_intervals']))
    summary['empirical_relative_changes']={k:emp['empirical_Plambda']['final']['192'][k]/emp['empirical_P0']['final']['192'][k]-1 for k in ['KL','TV','q_ratio','field_ratio','b_RMSE']}
    summary['interpretation_boundary']='Relative suppression of an expanded empirical branch is not P1-F near-zero collapse or teacher-relative shrinkage. Numeric failure and nonstationarity retained.'
    write(out/'SUMMARY.json',summary)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':130})
    def save(fig,name):fig.tight_layout();fig.savefig(out/'figures'/name);plt.close(fig)
    names=list(emp)
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for k in names:
        h=hist[k];x=[v['step'] for v in h]
        for ax,key in zip(axs,['empirical_ppp','total_objective','penalty_total']):ax.plot(x,[v[key] for v in h],label=k);ax.set_ylabel(key);ax.legend()
    axs[-1].set_xlabel('Optimizer updates');save(fig,'empirical_objectives.png')
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for k in names:
        for ax,key in zip(axs,['KL','TV']):ax.plot([v['step'] for v in obs[k]],[max(1e-15,v['96'][key]) for v in obs[k]],label=k);ax.set_yscale('log');ax.set_ylabel(key+' (96 grid)');ax.legend()
    axs[-1].set_xlabel('Optimizer updates');save(fig,'independent_recovery.png')
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for k in names:
        for ax,key in zip(axs,['q_ratio','field_ratio']):ax.plot([v['step'] for v in obs[k]],[v['96'][key] for v in obs[k]],label=k);ax.set_ylabel(key+' (96 grid)');ax.legend()
    for ax in axs:ax.axhline(1,color='gray',linestyle=':')
    axs[-1].set_xlabel('Optimizer updates');save(fig,'functional_contributions.png')
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for k in names:
        h=hist[k];x=[v['step'] for v in h]
        for ax,vals,label in [(axs[0],[v['u_norm'] for v in h],'u norm'),(axs[1],[v['h']['RMS'] for v in h],'h RMS'),(axs[2],[v['q']['RMS'] for v in h],'q RMS')]:ax.plot(x,vals,label=k);ax.set_ylabel(label);ax.legend()
    axs[-1].set_xlabel('Optimizer updates');save(fig,'parameter_compensation.png')
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for k in allruns:
        h=hist[k]
        for ax,key in zip(axs[:2],['gradient','actual_update']):ax.plot([v['step'] for v in h[1:]],[max(v[key]['all'],1e-15) for v in h[1:]],label=k,alpha=.8);ax.set_yscale('log');ax.set_ylabel(key+' L2');ax.legend(fontsize=8)
    for k in names:axs[2].plot([v['step'] for v in hist[k][1:]],[v['relative_update_norm'] for v in hist[k][1:]],label=k)
    axs[2].set_ylabel('Relative actual update');axs[2].legend();axs[-1].set_xlabel('Optimizer updates');save(fig,'optimization_dynamics.png')
    fig,axs=plt.subplots(2,3,figsize=(13,8));keys=['KL','TV','centered_g_area_RMSE','q_ratio','field_ratio','b_RMSE'];labels=['Pop 0','Pop original','Emp 0','Emp original']
    for ax,key in zip(axs.flat,keys):
        vals=[v['final']['192'][key] for v in allruns.values()];ax.bar(labels,vals,color=['#4773aa','#86a7ce','#c65c40','#e99c82']);ax.set_title(key+' @ step2000, grid192');ax.tick_params(axis='x',rotation=20)
        if key in ['KL','TV','centered_g_area_RMSE']:ax.set_yscale('log')
    fig.suptitle('Empirical endpoints fail the 96/192 logZ gate: numerical values are provisional',fontsize=11)
    save(fig,'factorial_endpoints.png')
    # Every predeclared flag is preserved in JSON. Show a fixed +/-50 step window around each run's first flagged event.
    for k in names:
        flagged=sorted(set(dyn[k]['objective_spike']+dyn[k]['gradient_spike']+dyn[k]['transition_candidate']))
        if not flagged:continue
        center=flagged[0];h=[v for v in hist[k] if abs(v['step']-center)<=50];fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
        for ax,key in zip(axs,['empirical_ppp','q_ratio','field_ratio']):ax.plot([v['step'] for v in h],[v[key] for v in h]);ax.axvline(center,color='red',linestyle=':');ax.set_ylabel(key)
        axs[0].set_title(k+f': first diagnostic flag at {center}; not proof of basin change');save(fig,k+'_transition_window.png')
    fig,axs=plt.subplots(4,1,figsize=(10,10),sharex=True)
    for k in names:
        for ax,key in zip(axs[:2],['total_objective','empirical_ppp']):ax.plot([v['step'] for v in obs[k]],[v[key] for v in obs[k]],label=k);ax.set_ylabel(key);ax.legend()
        for ax,key in zip(axs[2:],['KL','centered_g_area_RMSE']):ax.plot([v['step'] for v in obs[k]],[v['96'][key] for v in obs[k]],label=k);ax.set_ylabel(key+' (96)');ax.legend()
    axs[-1].set_xlabel('Optimizer updates');save(fig,'objective_conflict.png')
    notes=['# P1-G fixed 2x2 evidence','', 'All endpoints are fixed step2000, not selected checkpoints. KL/g metrics use grid192. Functional collapse and density recovery are separate.','', '| objective / penalty | KL | q ratio | field ratio | collapse ever | stationarity | fidelity |','|---|---:|---:|---:|---|---|---|']
    for k,v in summary['runs'].items():notes.append(f'| {k} | {v["KL"]:.9g} | {v["q_ratio"]:.9g} | {v["field_ratio"]:.9g} | {v["collapse_ever"]} | {v["stationarity"]["passed"]} | {v["fidelity"]["passed"]} |')
    notes+=['','## Interaction','| metric | empirical effect | population effect | interaction |','|---|---:|---:|---:|']
    for k,v in interaction.items():notes.append(f'| {k} | {v["empirical_effect"]:.9g} | {v["population_effect"]:.9g} | {v["interaction"]:.9g} |')
    notes+=['','## Objective conflict',json.dumps(summary['conflict'],indent=2),'','Full intervals, deltas and transition steps: TRAJECTORY_DYNAMICS.json. No significance estimate; one dataset/init. Fidelity failure limits continuous-domain interpretation. All saved states are descriptive; no restart/extension.']
    interpretation='''# P1-G 中文判读

1. empirical λ=0：截至2000步，逐步48²及每25步96²监测均未观察到P1-F定义的collapse；密度/预测恢复不通过。
2. empirical 原λ：同样未观察到collapse；密度/预测恢复不通过。
3. 正则效应与population不同：192²终点empirical KL下降约47.93%，q幅度下降50.54%，field contribution下降41.51%；这些是相对无正则膨胀支路的压低，原λ的q/teacher仍约3.54、field/teacher仍约2.33，并非相对teacher塌缩。population中KL效应为+4.0246e-5，empirical为-0.0812884，interaction=-0.0813287。无统计显著性含义。
4. 本轮没有预声明的objective conflict：未出现total下降、经验数据项上升、KL和centered-g误差共同上升的相邻步或独立监测区间，持续条件也未触发。反而，经验数据项与total共同改善而teacher恢复变差的96²区间，在P0/原λ分别有46/36个；这与Case F不同。
5. 当前最支持有限样本经验目标拟合与teacher恢复脱钩，伴随优化振荡和积分分辨率问题。λ不是坏恢复的必要条件，原λ在固定终点减轻了函数幅度扩张，但未恢复teacher，也未消除优化不稳定。不能把它写成正则诱导collapse、已证明结构性不可辨识性或模型bug。

优化动态并未随终点KL改善而全面改善。最大gradient为P0=2320.57、原λ=2978.12；预声明abs(单步total变化)>1的次数为273/353。原λ的gradient/n>10或相对尖峰条件触发2步；P0虽无该阈值标志，绝对梯度仍很大，不能说稳定。实际更新量另列，不能用梯度大小替代Adam的真实步长。候选transition不是已证明的basin改变。

## 必须保留的数值限制

两个empirical终点都没有通过96²→192² logZ绝对误差≤1e-4的门槛：P0为0.000602173，原λ为0.000112211。KL/TV的对应一致性条件通过，但不能以此豁免logZ失败。48²→192²目标修正分别约+0.844975/+0.164026。这是数值未解决，不是执行失败；没有放宽门槛或重训。报告的192²值及interaction是当前离散评价结果，不能当作已验证的连续域精确量。

全部四个2×2终点都未通过沿用的末200步stationarity，不宣称任何一个是目标最优解。λ=0/原λ的empirical末段max gradient/256约9.065/11.633，total范围/256约0.03049/0.04224。

Case A/B（collapse）不获支持；Case D（原λ恢复更差）与本轮固定终点方向相反；Case E不能描述明显的功能输出差异；Case F未触发。只符合Case C中“本设置未复现collapse”这部分，不符合“两条恢复正常”，因此不强行贴完整Case C。历史24²/.01的collapse仍未解释。

## 唯一下一步建议（未执行）

固定本轮两份step2000模型，不训练、不改参数，只做192²与384²的独立积分一致性检查，沿用原数值容差；先辨别终点评价误差，随后再决定是否能解释优化/正则机制。本轮没有执行384²或任何新训练。

'''
    (out/'NOTES.md').write_text(interpretation+'\n'.join(notes)+'\n',encoding='utf-8')
    numerical={}
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for k in names:
        seq=obs[k];gaps=[v['logZ48_96_abs'] for v in seq]
        numerical[k]=dict(endpoint96_192=emp[k]['fidelity'],training48_endpoint192_logZ=emp[k]['training_to_endpoint_logZ'],empirical48=emp[k]['empirical_ppp'],empirical192=emp[k]['empirical_ppp192'],max_logZ48_96=max(gaps),first_monitor_gap_above_1e_minus4=next((v['step'] for v in seq if v['logZ48_96_abs']>1e-4),None),note='48/96 discrepancy is descriptive, not a newly introduced stopping gate. Endpoint96/192 uses unchanged P1-F tolerance.')
        axs[0].plot([v['step'] for v in seq],[max(x,1e-16) for x in gaps],label=k)
        axs[1].plot([v['step'] for v in seq],[v['empirical_ppp_96']-v['empirical_ppp_48'] for v in seq],label=k)
    axs[0].set_yscale('log');axs[0].set_ylabel('abs(logZ48 - logZ96)');axs[0].legend();axs[1].set_ylabel('Empirical96 - empirical48');axs[1].legend();axs[1].set_xlabel('Optimizer updates');save(fig,'quadrature_discrepancy.png')
    write(out/'NUMERICAL_FIDELITY.json',numerical)
    write(out/'REPORT_PROVENANCE.json',dict(source=Path(__file__).name,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_end_sha256=hashlib.sha256((out/'TRAINING_END.json').read_bytes()).hexdigest()))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':report(REPO/read(REPO/'docs/project_memory/P1G_SESSION.json')['output_dir'])
