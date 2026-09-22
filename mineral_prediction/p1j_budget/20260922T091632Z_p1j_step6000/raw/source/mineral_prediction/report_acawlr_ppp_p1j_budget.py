"""P1-J budget comparison and equal-block dynamics; no model execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;MEM=REPO/'docs/project_memory'
OLD=ROOT/'p1i_learning_rate/20260922T082159Z_p1i_lr001';NAMES=['empirical_P0','empirical_Plambda'];STAGES=[2000,4000,6000]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def counts(rr):
    return dict(reversals=sum(x['linear']['reversal'] for x in rr),positive_predictions=sum(x['linear']['prediction']>1e-8 for x in rr),actual_increases=sum(x['linear']['actual']>1e-8 for x in rr),positive_prediction_and_increase=sum(x['linear']['prediction']>1e-8 and x['linear']['actual']>1e-8 for x in rr))
def monitor_conflicts(o,a,b):
    vv=[x for x in o if a<=x['step']<=b];records=[]
    for x,y in zip(vv,vv[1:]):
        d=dict(total=y['total_objective']-x['total_objective'],data=y['empirical_ppp']-x['empirical_ppp'],KL=y['192']['KL']-x['192']['KL'],g=y['192']['centered_g_area_RMSE']-x['192']['centered_g_area_RMSE']);records.append(dict(start=x['step'],end=y['step'],delta=d,overfit=d['total']<-1e-8 and d['data']<-1e-8 and d['KL']>1e-8 and d['g']>1e-8,conflict=d['total']<-1e-8 and d['data']>1e-8 and d['KL']>1e-8 and d['g']>1e-8))
    return records

def report(out):
    ep=read(out/'ENDPOINTS.json');di=read(out/'ENDPOINT_DIAGNOSTICS.json');hist={};obs={};summary={};mon={}
    for label in NAMES:
        a=rows(OLD/'raw'/f'{label}_trajectory.jsonl');b=rows(out/'raw'/f'{label}_trajectory.jsonl');assert a[-1]==b[0];hist[label]=a+b[1:];aa=read(OLD/'raw'/f'{label}_observations.json');bb=read(out/'raw'/f'{label}_observations.json');assert aa[-1]==bb[0];obs[label]=aa+bb[1:]
    for stage in STAGES:
        for label in NAMES:
            key=f'step{stage}_{label}';h=hist[label];block=h[stage-1999:stage+1];tail=h[stage-199:stage+1];m=ep[key]['metrics']['768'];fid=ep[key]['fidelity']['passed'];cc=counts(block);gmax=max(v['gradient']['all'] for v in tail)/256;span=(max(v['total_objective'] for v in tail)-min(v['total_objective'] for v in tail))/256;mc=monitor_conflicts(obs[label],stage-2000,stage);mon[key]=mc
            summary[key]=dict(budget=stage,learning_rate=.001,training_grid=96,metrics768=m,fidelity=ep[key]['fidelity'],stationarity=dict(max_gradient_per256=gmax,total_range_per256=span,passed=gmax<=1e-3 and span<=1e-5),block_start=stage-1999,block_end=stage,block_counts=cc,block_rates={k:v/2000 for k,v in cc.items()},objective_spikes=sum(v['flags']['objective_spike'] for v in block),gradient_max=max(v['gradient']['all'] for v in block),collapse_ever48=any(v['labels']['collapse'] for v in h[:stage+1]),density_success=fid and m['labels']['density_success'],predictor_success=fid and m['labels']['predictor_success'],empirical96=h[stage]['empirical_ppp'],total96=h[stage]['total_objective'],overfit_intervals192=sum(v['overfit'] for v in mc),strict_conflicts192=sum(v['conflict'] for v in mc))
    metrics=['KL','TV','centered_g_area_RMSE','centered_g_teacher_RMSE','b_RMSE','q_ratio','field_ratio','beta_error','u_norm','theta_norm','empirical_ppp'];contrast={}
    for n in ['384','768']:
        contrast[n]={}
        for a,b in [(2000,4000),(2000,6000),(4000,6000)]:
            contrast[n][f'{a}_to_{b}']={label:{k:dict(old=ep[f'step{a}_{label}']['metrics'][n][k],new=ep[f'step{b}_{label}']['metrics'][n][k],difference=ep[f'step{b}_{label}']['metrics'][n][k]-ep[f'step{a}_{label}']['metrics'][n][k],relative_change=ep[f'step{b}_{label}']['metrics'][n][k]/ep[f'step{a}_{label}']['metrics'][n][k]-1 if ep[f'step{a}_{label}']['metrics'][n][k] else None) for k in metrics} for label in NAMES}
    historical={label:read(OLD/'ENDPOINTS.json')['lr0.003_'+label]['metrics']['768'] for label in NAMES}
    write(out/'COMPARISON.json',contrast);write(out/'TRAJECTORY_DYNAMICS.json',dict(independent192_intervals=mon,blocks={k:{n:v[n] for n in ['block_start','block_end','block_counts','block_rates','objective_spikes','gradient_max']} for k,v in summary.items()}));write(out/'SUMMARY.json',dict(run_id=out.name,new_updates=8000,resume_step=2000,end_step=6000,temporary_displacements=2,runs=summary,historical_lr003_2000=historical,all_endpoint_fidelity=all(x['fidelity']['passed'] for x in summary.values()),interpretation='Budget intervention, not lr.003x2000 equivalence. No best checkpoint, statistical significance or optimum claim.'))
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':140})
    def save(fig,name):fig.tight_layout();fig.savefig(out/'figures'/name);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for label,o in obs.items():
        for ax,k in zip(axs.flat,['KL','b_RMSE','q_ratio','field_ratio']):ax.plot([v['step'] for v in o],[v['192'][k] for v in o],label=label);ax.set_ylabel(k+' on192');ax.legend()
    for ax in axs.flat:
        for s in [2000,4000]:ax.axvline(s,color='gray',ls='--',lw=.8)
    for ax in axs[-1]:ax.set_xlabel('Cumulative updates (resume at2000)')
    save(fig,'full_independent_monitor192.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for label,h in hist.items():
        x=[v['step'] for v in h]
        for ax,k in zip(axs[0],['empirical_ppp','total_objective']):ax.plot(x,[v[k] for v in h],label=label);ax.set_ylabel(k+' on96');ax.legend()
        axs[1,0].plot(x,[v['gradient']['all']/256 for v in h],label=label);axs[1,1].plot(x,[v['actual_update']['all'] for v in h],label=label)
    axs[1,0].set_ylabel('Gradient norm /256');axs[1,1].set_ylabel('Actual update norm')
    for ax in axs.flat:
        for s in [2000,4000]:ax.axvline(s,color='gray',ls='--',lw=.8)
    for ax in axs[1]:ax.set_xlabel('Cumulative updates');ax.legend()
    save(fig,'full_objective_dynamics.png')
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for label in NAMES:
        for ax,k in zip(axs,['KL','b_RMSE','q_ratio']):ax.plot(STAGES,[summary[f'step{s}_{label}']['metrics768'][k] for s in STAGES],marker='o',label=label);ax.set_ylabel(k+' on768');ax.set_xticks(STAGES);ax.set_xlabel('Fixed budget');ax.legend(fontsize=8)
    save(fig,'fixed_budget_endpoints768.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for label in NAMES:
        axs[0].plot(STAGES,[summary[f'step{s}_{label}']['block_rates']['reversals'] for s in STAGES],marker='o',label=label);axs[1].plot(STAGES,[summary[f'step{s}_{label}']['objective_spikes'] for s in STAGES],marker='o',label=label)
    axs[0].set_ylabel('Sign reversal rate per2000 updates');axs[1].set_ylabel('Objective spike count per2000 updates')
    for ax in axs:ax.set_xlabel('End of equal2000-update block');ax.set_xticks(STAGES);ax.legend()
    save(fig,'equal_block_dynamics.png')
    lines=['# P1-J：固定lr=.001，从2000步续训至6000步','','保留模型和完整Adam状态，两个lambda各新增4000步，共8000更新。固定96²、data23/init1011、teacher与全部优化器/正则设置。第2000步为归档边界，第2001步才是新增更新。第4000及6000步事前固定评价，不选最佳checkpoint。','', '## 统一384/768评价的六个固定状态','','| 步数/λ | KL768 | TV768 | g面积RMSE | b RMSE | q比例 | 场贡献比例 | fidelity | stationarity |','|---|---:|---:|---:|---:|---:|---:|---|---|']
    for key,s in summary.items():
        m=s['metrics768'];lines.append(f"| {key} | {m['KL']:.10g} | {m['TV']:.8g} | {m['centered_g_area_RMSE']:.8g} | {m['b_RMSE']:.8g} | {m['q_ratio']:.8g} | {m['field_ratio']:.8g} | {s['fidelity']['passed']} | {s['stationarity']['passed']} |")
    lines+=['','p是归一化密度，g是标量预测，b=beta+u*h是向量场，theta是神经参数。q及场贡献比例相对teacher；不将相对抑制称为近零collapse。','', '## 延长预算的配对变化（768²）','','| 区间/λ | KL变化 | KL相对变化 | b RMSE变化 | 经验PPP变化 |','|---|---:|---:|---:|---:|']
    for a,b in [(2000,4000),(2000,6000),(4000,6000)]:
        for label in NAMES:
            c=contrast['768'][f'{a}_to_{b}'][label];lines.append(f"| {a}→{b}/{label} | {c['KL']['difference']:.9g} | {c['KL']['relative_change']:.3%} | {c['b_RMSE']['difference']:.9g} | {c['empirical_ppp']['difference']:.9g} |")
    lines+=['','## 等长更新区间与stationarity','','反转指g_prev·实际位移<−1e-8、实际Δtotal>1e-8；预测上升>1e-8，实际上升>1e-8。分母均2000。目标突变为全局25步后absΔtotal>1，续训不重置检测历史。','', '| 步数区间/λ | 反转 | 预测上升 | 实际上升 | 目标突变 | 末200行max梯度/256 | 末200行total range/256 |','|---|---:|---:|---:|---:|---:|---:|']
    for key,s in summary.items():
        c=s['block_counts'];st=s['stationarity'];lines.append(f"| {s['block_start']}–{s['block_end']}/{key.split('_',1)[1]} | {c['reversals']} | {c['positive_predictions']} | {c['actual_increases']} | {s['objective_spikes']} | {st['max_gradient_per256']:.8g} | {st['total_range_per256']:.8g} |")
    lines+=['','stationarity沿用max梯度/256≤1e-3且total range/256≤1e-5，不能将value fidelity通过叫已收敛。','', '## 独立192监测区间','','| 步数区间/λ | 经验项及total下降但KL/g误差上升 | 严格Case F冲突 |','|---|---:|---:|']
    for key,s in summary.items():lines.append(f"| {s['block_start']}–{s['block_end']}/{key.split('_',1)[1]} | {s['overfit_intervals192']} | {s['strict_conflicts192']} |")
    lines+=['','每个区间80对相邻25步监测。Case F严格要求total下降但经验项、KL及g误差上升。','', '## 最终6000步的局部诊断','','| λ | 梯度96/384相对差 | 夹角° | 网格 | 候选一阶Δtotal | 实际Δtotal | 反转 |','|---|---:|---:|---|---:|---:|---|']
    for label,d in di.items():
        c=d['gradient_comparison']['total']['all']
        for n in ['96','384']:
            v=d['grids'][n]['changes']['total'];lines.append(f"| {label} | {c['relative_difference']:.8g} | {c['angle_degrees']:.8g} | {n} | {v['prediction']:.9g} | {v['actual']:.9g} | {v['reversal']} |")
    lines+=['','仅各一份临时副本完整候选位移，optimizer更新0，不是第6001步训练；384导数收敛尚未证明。','', '## 解释边界与完整证据','','延长预算不是与lr=.003×2000的自动等价进度。更低经验目标不保证teacher恢复，未达stationarity时不把剩余误差叫不可约统计误差或正则偏差。单数据、单初始化，不作总体显著或历史collapse因果结论。保留所有失败门槛，不继续预算或网格扫描。','', '完整指标ENDPOINTS.json，配对差异COMPARISON.json，等长区间TRAJECTORY_DYNAMICS.json；验证见FINAL_VERIFY.json和EVALUATION_VERIFY.json。SOURCE_MANIFEST及DELIVERY_SOURCE_MANIFEST记录冻结代码；旧状态及结果只读。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:dict(KL=v['metrics768']['KL'],fidelity=v['fidelity']['passed'],stationarity=v['stationarity']['passed'],reversals=v['block_counts']['reversals']) for k,v in summary.items()}))
if __name__=='__main__':report(REPO/read(MEM/'P1J_SESSION.json')['output_dir'])
