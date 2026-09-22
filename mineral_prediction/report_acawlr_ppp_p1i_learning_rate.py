"""P1-I matched fixed-budget learning-rate report; no model execution."""
import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_acawlr_ppp_p1g_empirical import dynamics,conflict
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;MEM=REPO/'docs/project_memory'
OLD=ROOT/'p1h_quadrature/20260922T072637Z_p1h_train96';NAMES=['empirical_P0','empirical_Plambda']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def report(out):
    endpoints=read(out/'ENDPOINTS.json');diag=read(out/'ENDPOINT_DIAGNOSTICS.json');hist={};obs={};dyn={};conf={};summary={}
    for lr,folder in [(.003,OLD),(.001,out)]:
        for label in NAMES:
            key=f'lr{lr:.3f}_{label}';v=rows(folder/'raw'/f'{label}_trajectory.jsonl');o=read(folder/'raw'/f'{label}_observations.json');hist[key]=v;obs[key]=o;dyn[key]=dynamics(v);oo=[dict(x,**{'96':x['192']}) for x in o];conf[key]=conflict(v,oo);conf[key]['scope']='All monitoring metrics are192 (helper keys inherited as96); same96 objective for both learning rates. Step diagnostics48.'
            m=endpoints[key]['metrics']['768'];gmax=max(x['gradient']['all'] for x in v[-200:])/256;span=(max(x['total_objective'] for x in v[-200:])-min(x['total_objective'] for x in v[-200:]))/256;fid=endpoints[key]['fidelity']['passed'];linear=[x['linear'] for x in v[1:]]
            counts=dict(reversals=sum(x['reversal'] for x in linear),positive_predictions=sum(x['prediction']>1e-8 for x in linear),actual_increases=sum(x['actual']>1e-8 for x in linear),positive_prediction_and_increase=sum(x['prediction']>1e-8 and x['actual']>1e-8 for x in linear))
            summary[key]=dict(learning_rate=lr,training_grid=96,metrics768=m,fidelity=endpoints[key]['fidelity'],stationarity=dict(max_gradient_per256=gmax,total_range_per256=span,passed=gmax<=1e-3 and span<=1e-5),objective_spikes=dyn[key]['objective_spike_count'],gradient_max=dyn[key]['gradient_max'],collapse_ever48=bool(dyn[key]['collapse_indicator']),density_success=fid and m['labels']['density_success'],predictor_success=fid and m['labels']['predictor_success'],linear_counts=counts,linear_rates={k:n/2000 for k,n in counts.items()},reversal_steps=[x['step'] for x in v[1:] if x['linear']['reversal']],empirical96=v[-1]['empirical_ppp'],total96=v[-1]['total_objective'],overfit_intervals192=len(conf[key]['overfit_monitor_intervals']),strict_conflicts192=len(conf[key]['monitor96_conflicts']))
    metrics=['KL','TV','centered_g_area_RMSE','centered_g_teacher_RMSE','b_RMSE','q_ratio','field_ratio','beta_error','u_norm','theta_norm','empirical_ppp'];contrasts={}
    for n in ['384','768']:
        contrasts[n]=dict(lr_effect={label:{k:dict(old=endpoints['lr0.003_'+label]['metrics'][n][k],new=endpoints['lr0.001_'+label]['metrics'][n][k],difference=endpoints['lr0.001_'+label]['metrics'][n][k]-endpoints['lr0.003_'+label]['metrics'][n][k],relative_change=endpoints['lr0.001_'+label]['metrics'][n][k]/endpoints['lr0.003_'+label]['metrics'][n][k]-1 if endpoints['lr0.003_'+label]['metrics'][n][k] else None) for k in metrics} for label in NAMES},lambda_effect={f'{lr:.3f}':{k:endpoints[f'lr{lr:.3f}_empirical_Plambda']['metrics'][n][k]-endpoints[f'lr{lr:.3f}_empirical_P0']['metrics'][n][k] for k in metrics} for lr in [.003,.001]})
    write(out/'COMPARISON.json',contrasts);write(out/'TRAJECTORY_DYNAMICS.json',dict(dynamics=dyn,independent192_conflict=conf));write(out/'SUMMARY.json',dict(run_id=out.name,new_trajectories=2,new_updates=4000,temporary_displacements=2,runs=summary,all_endpoint_fidelity=all(v['fidelity']['passed'] for v in summary.values()),interpretation='Same96 empirical objective and fixed2000 budget. Lower lr can slow fitting; recovery improvement is not optimized-minimum or stationarity evidence. Single dataset/init, no significance or lr sweep.'))
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':140})
    def save(fig,name):fig.tight_layout();fig.savefig(out/'figures'/name);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for key,o in obs.items():
        for ax,k in zip(axs.flat,['KL','b_RMSE','q_ratio','field_ratio']):ax.plot([v['step'] for v in o],[v['192'][k] for v in o],label=key.replace('empirical_',''));ax.set_ylabel(k+' on192');ax.legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Updates')
    save(fig,'independent_monitor192.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for key,v in hist.items():
        x=[a['step'] for a in v];name=key.replace('empirical_','')
        for ax,k in zip(axs[0],['empirical_ppp','total_objective']):ax.plot(x,[a[k] for a in v],label=name);ax.set_ylabel(k+' on96');ax.legend(fontsize=8)
        axs[1,0].plot(x,[a['gradient']['all']/256 for a in v],label=name);axs[1,1].plot(x,[a['actual_update']['all'] for a in v],label=name)
    axs[1,0].set_ylabel('Gradient norm /256');axs[1,1].set_ylabel('Actual update norm')
    for ax in axs[1]:ax.set_xlabel('Updates');ax.legend(fontsize=8)
    save(fig,'same_objective_dynamics.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,(key,v) in zip(axs.flat,hist.items()):
        a=v[1:];ax.scatter([x['linear']['prediction'] for x in a],[x['linear']['actual'] for x in a],s=4,c=['#d62728' if x['linear']['reversal'] else '#1f77b4' for x in a],alpha=.4);ax.axhline(0,color='black',lw=.6);ax.axvline(0,color='black',lw=.6);ax.set_xlabel('Previous gradient dot actual update');ax.set_ylabel('Actual total change96');ax.set_title(key.replace('empirical_',''))
    save(fig,'four_trajectory_linearity.png')
    fig,axs=plt.subplots(1,3,figsize=(13,4));keys=list(summary);labels=['.003:0','.003:lambda','.001:0','.001:lambda']
    for ax,k in zip(axs,['KL','b_RMSE','q_ratio']):ax.bar(labels,[summary[v]['metrics768'][k] for v in keys],color=['#8fb6d9','#dba48c','#387db8','#b95437']);ax.set_ylabel(k+' on768');ax.set_xlabel('Learning rate : penalty')
    save(fig,'four_endpoints768.png')
    lines=['# P1-I：固定96²训练，仅将学习率.003改为.001','','归档data23/init1011、teacher、float64 CPU1线程、Adam其他设置、lambda0/原lambda、2000步预算均保持不变。新增2条轨迹共4000更新；没有调参扫描、提前停止或最佳checkpoint选择。','', '## 统一终点结果（768²）','','| 学习率/λ | KL | TV | g面积RMSE | b RMSE | q比例 | 场贡献比例 | fidelity384/768 | stationarity |','|---|---:|---:|---:|---:|---:|---:|---|---|']
    for key,s in summary.items():
        m=s['metrics768'];lines.append(f"| {key} | {m['KL']:.10g} | {m['TV']:.8g} | {m['centered_g_area_RMSE']:.8g} | {m['b_RMSE']:.8g} | {m['q_ratio']:.8g} | {m['field_ratio']:.8g} | {s['fidelity']['passed']} | {s['stationarity']['passed']} |")
    lines+=['','q为空间预测分支，g为标量预测，p为归一化密度；b=beta+u*h为向量场。q及场贡献比例相对teacher。p恢复不等于b或参数theta可识别；不将相对抑制称为近零collapse。','', '## 学习率干预差异','','| λ | KL变化.001−.003 | KL相对变化 | b RMSE变化 | 经验PPP768变化 |','|---|---:|---:|---:|---:|']
    for label in NAMES:
        c=contrasts['768']['lr_effect'][label];lines.append(f"| {label} | {c['KL']['difference']:.10g} | {c['KL']['relative_change']:.3%} | {c['b_RMSE']['difference']:.10g} | {c['empirical_ppp']['difference']:.10g} |")
    lines+=['','固定预算学习率干预改变整个训练路径。较小lr下teacher指标更好，可能同时伴随经验目标更高，即拟合较慢；不能直接当作找到更好的经验最优解，也不据此延长训练补齐进度。','', '## 真实更新与stationarity','','全部率的分母为2000更新。反转：预测<-1e-8、实际Δtotal>1e-8；预测上升：g_prev·d_actual>1e-8；实际上升：Δtotal>1e-8。两轮目标与记录定义相同，可以配对比较；这不是逐步384梯度核验。','', '| 学习率/λ | 反转次数 | 预测上升次数 | 实际上升次数 | 目标突变次数 | 末200行max梯度/256 | 末200行total range/256 |','|---|---:|---:|---:|---:|---:|---:|']
    for key,s in summary.items():
        c=s['linear_counts'];st=s['stationarity'];lines.append(f"| {key} | {c['reversals']} | {c['positive_predictions']} | {c['actual_increases']} | {s['objective_spikes']} | {st['max_gradient_per256']:.8g} | {st['total_range_per256']:.8g} |")
    lines+=['','stationarity沿用末200行max梯度/256≤1e-3且total range/256≤1e-5；目标突变为25步后absΔtotal>1。','', '## 独立192监测','','| 学习率/λ | total及经验项下降但KL/g误差上升的区间数 | 严格Case F冲突区间数 |','|---|---:|---:|']
    for key,s in summary.items():lines.append(f"| {key} | {s['overfit_intervals192']} | {s['strict_conflicts192']} |")
    lines+=['','每条80个相邻25步监测区间。Case F严格要求total下降但经验项、KL及g误差均上升。','', '## 新终点梯度和候选完整位移','','| λ | 总梯度96/384相对差 | 夹角° | 网格 | 一阶Δtotal | 实际Δtotal | 反转 |','|---|---:|---:|---|---:|---:|---|']
    for label,d in diag.items():
        c=d['gradient_comparison']['total']['all']
        for n in ['96','384']:
            v=d['grids'][n]['changes']['total'];lines.append(f"| {label} | {c['relative_difference']:.8g} | {c['angle_degrees']:.8g} | {n} | {v['prediction']:.10g} | {v['actual']:.10g} | {v['reversal']} |")
    lines+=['','每个新终点只在临时副本施加一个完整候选位移，不调用optimizer.step，不是第2001步训练。384梯度未证明收敛，value fidelity不替代导数收敛。候选方向上升与一阶负、有限位移上升分开解释；不把两个不同状态间的差异当同状态正则直接干预。','', '## 完整证据','','核验结果见FINAL_VERIFY.json及EVALUATION_VERIFY.json；科学门槛失败与完整性失败分开。历史文件冻结、协议不回改、失败不重训。完整数据见ENDPOINTS.json、COMPARISON.json、TRAJECTORY_DYNAMICS.json与raw/。本轮只支持单数据单初始化固定预算比较，不作统计显著、全局最优或历史collapse原因结论。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:dict(KL=v['metrics768']['KL'],fidelity=v['fidelity']['passed'],stationarity=v['stationarity']['passed'],linear=v['linear_counts']) for k,v in summary.items()}))
if __name__=='__main__':report(REPO/read(MEM/'P1I_SESSION.json')['output_dir'])
