"""P1-F report from saved trajectories only. No model imports or optimization."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        key=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,key))
        elif not isinstance(v,(list,tuple)):out[key]=v
    return out

def compare_pair(p0,pl):
    a,b=p0['final']['192'],pl['final']['192'];valid=p0['fidelity']['passed'] and pl['fidelity']['passed']
    if not valid:case='numerical_unresolved'
    elif p0['density_success'] and p0['predictor_success'] and pl['stable_collapse']:case='A'
    elif p0['density_success'] and p0['predictor_success'] and pl['stable_shrinkage'] and b['KL']>a['KL']:case='D'
    elif p0['density_success'] and pl['density_success']:case='C'
    elif not p0['density_success'] and not pl['density_success']:case='B_or_unresolved'
    else:case='mixed_unresolved'
    return dict(case=case,fidelity=valid,delta_KL=b['KL']-a['KL'],q_ratio_P0=a['q_ratio'],q_ratio_Plambda=b['q_ratio'],
                delta_b_RMSE=b['b_RMSE']-a['b_RMSE'],density_P0=p0['density_success'],density_Plambda=pl['density_success'],
                stable_shrinkage=pl['stable_shrinkage'],stable_severe_shrinkage=pl['stable_severe_shrinkage'],stable_collapse=pl['stable_collapse'])

def dynamics(rows):
    passing=[r for r in rows if r['labels']['density_success']];first=passing[0] if passing else None;last=rows[-1]
    event=None if first is None else dict(first_density_success_step=first['step'],delta_total_to_final=last['total_objective']-first['total_objective'],
        delta_KL_to_final=last['KL']-first['KL'],delta_q_RMS_to_final=last['q']['RMS']-first['q']['RMS'],
        delta_penalty_to_final=last['penalty_total']-first['penalty_total'],scope='48-grid descriptive first threshold crossing, never checkpoint selection')
    intervals=[]
    for i,j in zip((0,100,300,600,1000,1500),(100,300,600,1000,1500,2000)):
        if j>=len(rows):continue
        a,b=rows[i],rows[j];intervals.append(dict(start=i,end=j,delta_total=b['total_objective']-a['total_objective'],delta_KL=b['KL']-a['KL'],
          delta_q_RMS=b['q']['RMS']-a['q']['RMS'],delta_penalty=b['penalty_total']-a['penalty_total']))
    count=sum(b['total_objective']<a['total_objective']-1e-10 and b['KL']>a['KL']+1e-10 and b['q']['RMS']<a['q']['RMS']-1e-10 for a,b in zip(rows,rows[1:]))
    return dict(first_threshold_event=event,fixed_intervals=intervals,adjacent_total_down_KL_up_q_down_steps=count)

def make_figures(out,results,hist):
    plt.rcParams.update({'font.size':10});colors={0.:'#2671b5',.1:'#24985f',1.:'#c44538'}
    specs={
      'objective_trajectories':[('total_objective','Total objective F'),('population_likelihood','256 Lpop (uncentered)'),('penalty_total','Effective penalty R')],
      'KL_trajectories':[('KL','KL, 48-grid'),('TV','TV, 48-grid'),('centered_g_area_RMSE','Centered g area RMSE')],
      'branch_contribution_trajectories':[('q_ratio','q RMS / teacher q RMS'),('u_norm','u norm'),('h.RMS','h area RMS'),('theta_norm','Neural parameter norm'),('gradient.theta','Neural gradient L2')],
      'penalty_decomposition':[('penalty_effective.beta','Effective beta penalty'),('penalty_effective.u','Effective u penalty'),('penalty_effective.theta','Effective neural penalty')]}
    for filename,panels in specs.items():
        fig,axes=plt.subplots(len(panels),2,figsize=(12,2.6*len(panels)),layout='constrained',squeeze=False)
        for col,start in enumerate(('standard','local')):
            for key,s in results.items():
                if not key.startswith(start+'_'):continue
                rows=[flatten(r) for r in hist[key]];x=[r['step'] for r in rows];lam=s['lambda_multiplier']
                for i,(metric,label) in enumerate(panels):
                    ax=axes[i,col];y=[r.get(metric,np.nan) for r in rows];ax.plot(x,y,color=colors[lam],label=f'lambda multiplier {lam:g}',lw=1);ax.set_ylabel(label);ax.grid(alpha=.2)
                    if metric in ('KL','gradient.theta','q_ratio','h.RMS'):ax.set_yscale('symlog',linthresh=1e-8)
                    if metric=='KL':ax.axhline(1e-4,color='gray',ls='--',lw=.7)
                    if metric=='q_ratio':ax.axhline(1.,color='gray',ls=':',lw=.7)
            axes[0,col].set_title(start+' initialization');axes[0,col].legend(fontsize=8);axes[-1,col].set_xlabel('Update step')
        fig.suptitle('P1-F matched population trajectories; all scheduled steps, no selection');fig.savefig(out/'figures'/f'{filename}.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained');keys=list(results);labels=[k.replace('standard','std').replace('Plambda','P1') for k in keys]
    for ax,(metric,label) in zip(axes.flat,[('KL','KL192'),('TV','TV192'),('q_ratio','q RMS / teacher192'),('b_RMSE','b-field RMSE192'),('beta_error','beta error192'),('centered_g_area_RMSE','Centered g RMSE192')]):
        ax.bar(np.arange(len(keys)),[results[k]['final']['192'][metric] for k in keys],color=[colors[results[k]['lambda_multiplier']] for k in keys]);ax.set_xticks(np.arange(len(keys)),labels,rotation=35,ha='right',fontsize=8);ax.set_title(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed step2000 endpoints, independent 192² evaluation');fig.savefig(out/'figures/P0_vs_Plambda_recovery.png',dpi=145);plt.close(fig)
    # Independent-grid KL monitor, to make training quadrature artifacts visible.
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for i,start in enumerate(('standard','local')):
        for key,s in results.items():
            if not key.startswith(start+'_'):continue
            obs=read(out/'raw'/f'{key}_observations.json');axes[i].plot([v['step'] for v in obs],[v['96']['KL'] for v in obs],label=f"lambda {s['lambda_multiplier']:g}",color=colors[s['lambda_multiplier']])
        axes[i].set(title=start+' independent 96² monitor',xlabel='Update step',ylabel='KL96',yscale='symlog');axes[i].set_yscale('symlog',linthresh=1e-10);axes[i].axhline(1e-4,color='gray',ls='--');axes[i].legend();axes[i].grid(alpha=.2)
    fig.savefig(out/'figures/KL96_monitor.png',dpi=145);plt.close(fig)

def run(out):
    if (out/'SUMMARY.json').exists():raise FileExistsError('Do not overwrite a report')
    results=read(out/'POPULATION_COMPARISON.json');gates=read(out/'GATES.json');control=read(out/'CONTROLS.json');dose=read(out/'DOSE_DECISION.json');prot=read(out/'PROTOCOL.json')
    histories={k:[json.loads(s) for s in (out/'raw'/f'{k}_trajectory.jsonl').read_text(encoding='utf-8').splitlines()] for k in results}
    dynamics_all={k:dynamics(v) for k,v in histories.items()};pairs={s:compare_pair(results[s+'_P0'],results[s+'_Plambda']) for s in ('standard','local')}
    main_valid=all(results[k]['fidelity']['passed'] and results[k]['status']=='COMPLETED' for k in ('standard_P0','standard_Plambda','local_P0','local_Plambda'))
    p0passed=[s for s in ('standard','local') if results[s+'_P0']['density_success']];shrunk=[s for s in ('standard','local') if results[s+'_Plambda']['stable_shrinkage']]
    collapsed=[k for k,v in results.items() if v['stable_collapse']];anycollapsed=[k for k,v in results.items() if v['collapse_observed_steps']]
    dose_normal=[s for s in ('standard','local') if s+'_Psmall' in results and results[s+'_Psmall']['density_success'] and results[s+'_Psmall']['predictor_success'] and not results[s+'_Psmall']['stable_severe_shrinkage'] and not results[s+'_Psmall']['stable_collapse']]
    evidence='numerical_unresolved' if not main_valid else 'regularization_objective_bias' if any(p['fidelity'] and p['density_P0'] and p['stable_shrinkage'] and p['delta_KL']>0 for p in pairs.values()) else 'regularization_changes_recovery_without_stable_field_shrinkage' if any(p['fidelity'] and p['density_P0'] and p['delta_KL']>1e-6 for p in pairs.values()) else 'optimizer_or_parameterization_unresolved' if not p0passed else 'no_population_collapse_attribution'
    # Evidence grades are deliberately bounded by the actual sampled starts and dose response.
    if not main_valid:level='未定级（数值门槛未通过）';triage='终点积分不可靠时，不能把现象升格为结构病理。'
    elif dose['eligible'] and len(dose_normal)==2:level='Level 0 倾向';triage='0.1倍原正则在两个初态均达到预定恢复门槛，本轮更像强度/尺度问题，不支持独立方法论文。'
    elif len(shrunk)==2 and all(p['density_P0'] and p['delta_KL']>0 for p in pairs.values()):
        level='Level 2 候选，限本teacher与两个固定初态';triage='若两个初态均出现稳定功能收缩，支持有限范围内的方法学问题候选；尚无跨teacher/模型类、独立seed重复或原则性修复证据，远未达到Level3。'
    else:level='Level 1 上限或机制未决';triage='有理论正则偏差，但本轮不足以建立跨初始化稳定病理及历史collapse主因；优先作为第一篇的诊断证据。'
    if not main_valid:q1='P0 的恢复结论受终点数值门槛限制，不能定论。'
    else:q1='P0 达到预定密度恢复门槛的初态：'+('、'.join(p0passed) if p0passed else '无')+'；standard 与 local 分开报告，local 的保持成功不等于从远处恢复。'
    q2='Pλ 在末段满足预声明稳定收缩定义的初态：'+('、'.join(shrunk) if shrunk else '无')+'。'
    q3='观察到持续功能层面 collapse 的轨迹：'+('、'.join(collapsed) if collapsed else '无')+'；至少一次监测到 collapse 的轨迹：'+('、'.join(anycollapsed) if anycollapsed else '无')+'。'
    q4='当前证据判别：'+evidence+'。这是固定teacher/optimizer/初态下的匹配结果，不自动解释历史经验样本轨迹。'
    summary=dict(run_id=out.name,answers=[q1,q2,q3,q4],pairs=pairs,evidence=evidence,all_main_fidelity_passed=main_valid,trajectory_count=len(results),
        updates=sum(v['steps'] for v in results.values()),stage2=dose,small_lambda_normal_starts=dose_normal,second_paper_triage=dict(level=level,reason=triage),dynamics=dynamics_all,
        endpoints={k:dict(density_success=v['density_success'],predictor_success=v['predictor_success'],fidelity=v['fidelity'],stable_shrinkage=v['stable_shrinkage'],stable_collapse=v['stable_collapse'],metrics=v['final']['192'],local_probe=v['local_probe']) for k,v in results.items()})
    write(out/'SUMMARY.json',summary);write(out/'TRAJECTORY_DYNAMICS.json',dynamics_all)
    lines=['# P1-F 中文报告','',*['%d. %s'%(i+1,q) for i,q in enumerate(summary['answers'])],
      '', '## 实验与数值边界', '',
      f'本轮run：{out.name}。标准初态为正式归档init1011；local为seed20260918、相对范数0.001的teacher附近扰动。每条固定2000步、Adam lr=.003、float64/CPU1线程、48²训练/96²监测/192²终点评价；唯一配对差异为原显式L2的整体倍数。实际{len(results)}条、{summary["updates"]}次更新，无有限样本训练、真实数据或新方法。',
      '旧23/53/71是数据seed，在population目标下没有抽样作用，未当作三个独立初始化。P1-E旧gate与BLOCKED原样保留；本轮按真正依赖的teacher/controls/新endpoint检查fresh gate，并未放宽1e-4的logZ误差门槛。',
      '', '| 初态/条件 | KL192 | TV192 | centered g RMSE(面积/teacher) | b RMSE | beta误差 | q/teacher | 密度/预测通过 | 持续collapse |',
      '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for k,v in results.items():
        a=v['final']['192'];lines.append(f'| {k} | {a["KL"]:.8g} | {a["TV"]:.7g} | {a["centered_g_area_RMSE"]:.6g}/{a["centered_g_teacher_RMSE"]:.6g} | {a["b_RMSE"]:.6g} | {a["beta_error"]:.6g} | {a["q_ratio"]:.6g} | {v["density_success"]}/{v["predictor_success"]} | {v["stable_collapse"]} |')
    lines+=['','## 终点数值保真与stationarity','', '| 条件 | ΔlogZ96→192 | ΔKL96→192 | fidelity | grad/256末段最大 | total范围/256 | stationarity |', '|---|---:|---:|---|---:|---:|---|']
    for k,v in results.items():
        st=v['stationarity'];fi=v['fidelity'];lines.append(f'| {k} | {fi["errors"]["logZ"]:.7g} | {fi["errors"]["KL"]:.7g} | {fi["passed"]} | {st["max_gradient_per256"]:.6g} | {st["total_range_per256"]:.6g} | {st["passed"]} |')
    lines+=['', '固定2000步不早停。恢复门槛与stationarity分别报告；后者未通过时不宣称已经求得人口目标最优解。', '', '## 目标与参数惩罚', '', '| 条件 | beta penalty（原recipe） | u penalty | theta penalty | Fλ−Fλ(teacher),192² |', '|---|---:|---:|---:|---:|']
    for k,v in results.items():
        pp=v['penalty_reference'];lines.append(f'| {k} | {pp["beta"]:.8g} | {pp["u"]:.8g} | {pp["theta"]:.8g} | {v["penalized_excess_over_teacher_192"]:.8g} |')
    lines+=['', '上表惩罚分量使用原recipe作为共同评价尺度；P0训练实际惩罚为0，effective分量另存在逐步日志。正惩罚排除exact teacher为stationary点，并不排除通过改变参数表示仍达到小KL的近似密度恢复。不能把u的收缩直接解释成q或field塌缩。']
    for suffix in ('P0','Plambda'):
        left=results['standard_'+suffix]['penalized_excess_over_teacher_192'];right=results['local_'+suffix]['penalized_excess_over_teacher_192']
        lines.append(f'同一{suffix}目标下，local终点比standard终点的dense目标高{right-left:.8g}（负数表示local更低）。若一条轨迹高于另一已知可行点，说明该轨迹未达到已知更好目标，不能把其偏差全解释为目标最优点的性质。')
    lines+=['','## 轨迹是否把收缩与目标下降联系起来','', '以下仅描述固定轨迹的首次密度门槛交叉及预定checkpoint区间，未据此选择checkpoint或启动额外搜索。']
    for k,v in dynamics_all.items():
        event=v['first_threshold_event'];lines.append(f'- {k}：'+('从未达到48²密度恢复门槛。' if event is None else f'首次达到48²密度门槛在step{event["first_density_success_step"]}；此后到终点Δtotal={event["delta_total_to_final"]:.7g}，ΔKL={event["delta_KL_to_final"]:.7g}，ΔqRMS={event["delta_q_RMS_to_final"]:.7g}，Δpenalty={event["delta_penalty_to_final"]:.7g}。')+f' 相邻步同时total下降/KL上升/q下降的次数={v["adjacent_total_down_KL_up_q_down_steps"]}。')
    lines+=['','## 第一阶段判别与剂量实验','']
    for start,p in pairs.items():lines.append(f'- {start}：Case {p["case"]}；Pλ−P0的KL差={p["delta_KL"]:.8g}；q相对teacher由{p["q_ratio_P0"]:.6g}变为{p["q_ratio_Plambda"]:.6g}。')
    lines.append(f'预声明dose条件：{dose}。'+('仅追加两条0.1倍原正则，复用0与1的主轨迹；未加中间值或选最好λ。' if dose['eligible'] else '没有执行任何剂量训练。'))
    lines+=['','## 五层恢复与 local 初态的解释','',
      'p/g的门槛分别报告；b-field误差和beta误差不作为密度通过的替代。u/h的变化需结合q与field norm，不能因u减小就称collapse；theta距离只作内部表示描述，不能据此判科学失败。低KL也不能建立连续域field唯一性。']
    for name in ('local_P0','local_Plambda'):
        s=results[name];lines.append(f'{name}：初始KL192={s["initial192"]["KL"]:.9g}；最终KL192={s["final"]["192"]["KL"]:.9g}；局部probe={s["local_probe"]}。固定小扰动可能已满足绝对恢复门槛，必须检查excess是否降低，不把保持成功包装为冷启动恢复。')
    lines+=['','## 排除与未排除','',
      '本轮训练目标没有finite-sample噪声，因此实际观察到的population现象无需有限样本噪声即可发生。teacher完整state属于模型类，copy/梯度检查通过；使用固定同初态的匹配比较可以隔离“加入原正则”在这套离散优化程序中的因果影响。通过96/192只说明预定节点与容差内的数值可信度，不能证明连续积分完全精确。',
      '本轮没有复现历史data23/71的empirical目标，训练网格也由旧24²变为48²。因此即使观察到population收缩/塌缩，也只能证明候选机制可以发生，不能据此断言历史collapse主要由它导致。没有出现collapse同样不能排除别的初态、样本或优化设置下的collapse。',
      '本轮只有一个teacher、两个固定初态；没有collapse概率估计，没有参数/field不可辨识性的全局证明，没有跨模型类推广。stationarity只是末200步描述，低梯度不是全局最优证明。',
      '', '## SECOND_PAPER_TRIAGE','',level+'。'+triage,
      'Level3需要推广到一类模型、原则性修复及理论/实验检验；本轮明确未实现这些。Level2候选也不能等同于可发表结论。',
      '', '## 下一项最小实验','',
      ('先针对实际失败的新endpoint设计独立积分核验，保留本轮结果，不把误差解释为结构病理。' if not main_valid else
       '下一轮只做归档data23/init1011的empirical matched pair（λ=0与原λ），固定本轮相同48²/Adam/.003/2000步，和现有standard population pair比较；不扩seed。这个2×2目标/正则对照才开始检验收缩与历史经验collapse的联系。本轮未执行。'),
      '', '代码与证据入口：PROTOCOL.json、PROVENANCE.json、GATES.json、CONTROLS.json、POPULATION_COMPARISON.json、TRAJECTORIES.json、raw/及figures/。失败门槛和所有实际trajectory均保留。']
    (out/'NOTES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    (out/'DECISION.md').write_text('# P1-F decision\n\n'+q4+'\n\n'+str(dose)+'\n\nSECOND_PAPER_TRIAGE: '+level+'。'+triage+'\n\nNext experiment is only proposed in NOTES; no authorization inferred from this decision.\n',encoding='utf-8',newline='\n')
    make_figures(out,results,histories)
    write(out/'REPORT_PROVENANCE.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reader_only=True,input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'POPULATION_COMPARISON.json',out/'GATES.json',out/'DOSE_DECISION.json',out/'CONTROLS.json']}))
    print(json.dumps(dict(answers=summary['answers'],pairs=pairs,stage2=dose,second_paper=summary['second_paper_triage']),ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True,type=Path);run(p.parse_args().input_dir.resolve())
