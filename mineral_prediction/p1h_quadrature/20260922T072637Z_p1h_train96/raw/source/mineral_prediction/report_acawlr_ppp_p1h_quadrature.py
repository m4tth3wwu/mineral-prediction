"""P1-H descriptive paired comparison, no training or checkpoint selection."""
import json,math,statistics,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_acawlr_ppp_p1g_empirical import dynamics,conflict
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;MEM=REPO/'docs/project_memory'
OLD=ROOT/'p1g_empirical/20260922T054836Z_p1g_v1';NAMES=['empirical_P0','empirical_Plambda']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def report(out):
    endpoints=read(out/'ENDPOINTS.json');diag=read(out/'ENDPOINT_DIAGNOSTICS.json');hist={};obs={};dyn={};conf={};summary={}
    for n,folder in [(48,OLD),(96,out)]:
        for label in NAMES:
            key=f'train{n}_{label}';v=rows(folder/'raw'/f'{label}_trajectory.jsonl');o=read(folder/'raw'/f'{label}_observations.json');hist[key]=v;obs[key]=o;dyn[key]=dynamics(v);cc=conflict(v,o);cc['scope']='Objective uses each run training grid; step metrics48; monitor96 common but not independent for train96. No old/new per-step reversal-rate comparison.';conf[key]=dict(common96=cc)
            if n==96:
                oo=[dict(x,**{'96':x['192']}) for x in o];conf[key]['independent192']=conflict(v,oo);conf[key]['independent192']['scope']='Keys inherited from helper named96; here every monitor metric is192. Objective remains train96.'
            m=endpoints[key]['metrics']['768'];gmax=max(x['gradient']['all'] for x in v[-200:])/256;span=(max(x['total_objective'] for x in v[-200:])-min(x['total_objective'] for x in v[-200:]))/256;fid=endpoints[key]['fidelity']['passed']
            summary[key]=dict(training_grid=n,metrics768=m,fidelity=endpoints[key]['fidelity'],stationarity=dict(max_gradient_per256=gmax,total_range_per256=span,passed=gmax<=1e-3 and span<=1e-5),objective_spikes=dyn[key]['objective_spike_count'],gradient_max=dyn[key]['gradient_max'],collapse_ever48=bool(dyn[key]['collapse_indicator']),density_success=fid and m['labels']['density_success'],predictor_success=fid and m['labels']['predictor_success'],reversal_count=None if n==48 else sum(x['linear']['reversal'] for x in v[1:]),reversal_steps=None if n==48 else [x['step'] for x in v[1:] if x['linear']['reversal']])
    metrics=['KL','TV','centered_g_area_RMSE','centered_g_teacher_RMSE','b_RMSE','q_ratio','field_ratio','beta_error','u_norm','theta_norm','empirical_ppp'];contrasts={}
    for n in ['384','768']:
        contrasts[n]=dict(grid_effect={label:{k:dict(old=endpoints['train48_'+label]['metrics'][n][k],new=endpoints['train96_'+label]['metrics'][n][k],difference=endpoints['train96_'+label]['metrics'][n][k]-endpoints['train48_'+label]['metrics'][n][k],relative_change=endpoints['train96_'+label]['metrics'][n][k]/endpoints['train48_'+label]['metrics'][n][k]-1 if endpoints['train48_'+label]['metrics'][n][k] else None) for k in metrics} for label in NAMES},lambda_effect={str(t):{k:endpoints[f'train{t}_empirical_Plambda']['metrics'][n][k]-endpoints[f'train{t}_empirical_P0']['metrics'][n][k] for k in metrics} for t in [48,96]})
    write(out/'COMPARISON.json',contrasts);write(out/'TRAJECTORY_DYNAMICS.json',dict(dynamics=dyn,conflict=conf));write(out/'SUMMARY.json',dict(run_id=out.name,new_trajectories=2,new_updates=4000,temporary_displacements=2,runs=summary,all_endpoint_fidelity=all(v['fidelity']['passed'] for v in summary.values()),interpretation='Fixed dataset/init and fixed-horizon descriptive grid intervention. Nonstationary endpoints are not optima. No significance, dose/lr search, best-checkpoint, or past-collapse attribution.'))
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':140})
    def save(fig,name):fig.tight_layout();fig.savefig(out/'figures'/name);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for key,o in obs.items():
        for ax,k in zip(axs.flat,['KL','centered_g_area_RMSE','q_ratio','field_ratio']):ax.plot([v['step'] for v in o],[v['96'][k] for v in o],label=key.replace('empirical_',''));ax.set_ylabel(k+' on96');ax.legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Updates')
    fig.suptitle('Common monitor96 (training grid for train96 runs)');save(fig,'common_monitor96.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for label in NAMES:
        key='train96_'+label;o=obs[key];v=hist[key]
        for ax,k in zip(axs[0],['KL','b_RMSE']):ax.plot([a['step'] for a in o],[a['192'][k] for a in o],label=label);ax.set_ylabel(k+' on192');ax.legend()
        axs[1,0].plot([a['step'] for a in v],[a['total_objective'] for a in v],label=label);axs[1,1].plot([a['step'] for a in v],[a['gradient']['all']/256 for a in v],label=label)
    axs[1,0].set_ylabel('Total objective96');axs[1,1].set_ylabel('Gradient norm /256')
    for ax in axs[1]:ax.set_xlabel('Updates');ax.legend()
    save(fig,'new_training_and_independent192.png')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,label in zip(axs,NAMES):
        v=hist['train96_'+label][1:];x=[a['linear']['prediction'] for a in v];y=[a['linear']['actual'] for a in v];colors=['#d62728' if a['linear']['reversal'] else '#1f77b4' for a in v];ax.scatter(x,y,s=5,c=colors,alpha=.45);ax.axhline(0,color='black',lw=.6);ax.axvline(0,color='black',lw=.6);ax.set_xlabel('Previous gradient dot actual update');ax.set_ylabel('Actual total objective change96');ax.set_title(label+'; red = sign reversal')
    save(fig,'actual_update_linearity.png')
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    keys=list(summary);labels=['48:0','48:lambda','96:0','96:lambda']
    for ax,k in zip(axs,['KL','b_RMSE','q_ratio']):ax.bar(labels,[summary[v]['metrics768'][k] for v in keys],color=['#8fb6d9','#dba48c','#387db8','#b95437']);ax.set_ylabel(k+' on768');ax.set_xlabel('Training grid : penalty')
    save(fig,'four_endpoints768.png')
    lines=['# P1-H：训练积分96²与冻结48²的固定终点对照','','仅训练积分网格改变。归档data23/init1011、teacher、float64 CPU单线程、Adam lr=.003/betas=.9,.999/eps=1e-8、原正则系数与2000更新预算完全固定。新增lambda0/原lambda两条轨迹共4000更新；第2000步为唯一报告终点。','', '## 统一终点结果（768²）','','| 训练网格/λ | KL | TV | g面积RMSE | b RMSE | q比例 | 场贡献比例 | fidelity384/768 | stationarity |','|---|---:|---:|---:|---:|---:|---:|---|---|']
    for key,s in summary.items():
        m=s['metrics768'];lines.append(f"| {key} | {m['KL']:.10g} | {m['TV']:.8g} | {m['centered_g_area_RMSE']:.8g} | {m['b_RMSE']:.8g} | {m['q_ratio']:.8g} | {m['field_ratio']:.8g} | {s['fidelity']['passed']} | {s['stationarity']['passed']} |")
    lines+=['','q是空间预测分支，g是标量预测函数；归一化p由g确定。b=beta+u*h为向量场。q/场贡献比例均相对teacher。p较接近不等于b或参数theta可识别；不将分支相对抑制称为近零collapse。','', '## 网格干预的配对差异','','| λ | KL变化96−48 | KL相对变化 | b RMSE变化 | 384²上KL同方向 |','|---|---:|---:|---:|---|']
    for label in NAMES:
        c=contrasts['768']['grid_effect'][label];a=contrasts['384']['grid_effect'][label]['KL']['difference'];lines.append(f"| {label} | {c['KL']['difference']:.10g} | {c['KL']['relative_change']:.3%} | {c['b_RMSE']['difference']:.10g} | {a*c['KL']['difference']>0} |")
    lines+=['','## 轨迹与真实更新诊断','','目标突变使用既有定义：25步后abs(Δtotal)>1；各轨迹total使用自身训练积分网格。stationarity沿用末200行max||grad||/256≤1e-3且total range/256≤1e-5；并非最优性证明。','', '| 训练网格/λ | 目标突变次数 | 最大梯度范数 | 末200行max梯度/256 | 末200行total range/256 | 新记录反转次数/2000 |','|---|---:|---:|---:|---:|---:|']
    for key,s in summary.items():lines.append(f"| {key} | {s['objective_spikes']} | {s['gradient_max']:.8g} | {s['stationarity']['max_gradient_per256']:.8g} | {s['stationarity']['total_range_per256']:.8g} | {s['reversal_count'] if s['reversal_count'] is not None else '未记录'} |")
    lines+=['','真实更新反转指上一状态梯度点乘实际位移<−1e-8，但本次实际total变化>1e-8。逐步记录无额外更新。旧轨迹未存完整逐步梯度，不能比较旧/新的该事件率。监测96为公共比较网格，但对新轨迹不是独立评价；新轨迹另保存每25步192指标。','', '## 新终点的梯度与完整候选位移','','| λ | 梯度96/384相对差 | 夹角° | 网格 | 候选一阶Δtotal | 实际Δtotal | 反转 |','|---|---:|---:|---|---:|---:|---|']
    for label,d in diag.items():
        c=d['gradient_comparison']['total']['all']
        for n in ['96','384']:
            v=d['grids'][n]['changes']['total'];lines.append(f"| {label} | {c['relative_difference']:.8g} | {c['angle_degrees']:.8g} | {n} | {v['prediction']:.10g} | {v['actual']:.10g} | {v['reversal']} |")
    lines+=['','只按冻结第2000步Adam矩重建一个候选位移，并在每个新终点的临时副本加一次。两个探测不是第2001步训练，也未用于选择终点。384梯度和探测点未做更细导数收敛核验；余项不区分曲率与非光滑性，不把λ间不同终点的差异当同状态正则因果干预。','', '## 完整性与解释范围','','结果以FINAL_VERIFY.json与EVALUATION_VERIFY.json为准；完整性通过与科学恢复/fidelity门槛分开。所有旧文件冻结，失败门槛保留，不重训、不加网格或调参。','', '本轮只支持单数据、单初始化、固定预算的配对描述；不能推出总体统计显著、全局最优、历史collapse原因，或论文机制已被验证。当前结果应与P1-F population恢复良好、P1-G empirical恢复差的证据合读，不能靠本轮改写旧证据。','', '完整指标与差异：ENDPOINTS.json、COMPARISON.json；每步证据：raw/*_trajectory.jsonl；梯度/候选/探测原始数据：raw/*gradients*、*candidate*、*probe*；协议、输入与源码哈希已归档。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:dict(KL=v['metrics768']['KL'],fidelity=v['fidelity']['passed'],stationarity=v['stationarity']['passed'],reversals=v['reversal_count']) for k,v in summary.items()}))
if __name__=='__main__':report(REPO/read(MEM/'P1H_SESSION.json')['output_dir'])
