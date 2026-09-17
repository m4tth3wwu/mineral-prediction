"""Read-only LR comparison: all trajectories and descriptive extrema, no selection."""
import json
import statistics
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acawlr_ppp_p1d_lr import STEM, RATES, SEEDS, OSCILLATION
from acawlr_ppp_optimization import CHECKPOINTS, write_json


def summarize(case):
    h,obs=case['history'],case['observations']
    final=obs[-1]
    # Record exact collapse after step0; separately distinguish a branch that never grew.
    # Near-collapse initialization is not itself treated as a later collapse event.
    grown=False; collapses=[]; near=[]
    for o in obs:
        if o['h_std']>1e-4: grown=True
        if o['step']>0 and o['h_collapse']: collapses.append(o['step'])
        if grown and o['h_near_collapse']: near.append(o['step'])
    windows=[]
    for lo,hi in OSCILLATION['windows']:
        a=[x for x in h if lo<=x['step']<=hi]
        v=[x['ppp_objective'] for x in a]
        ran=max(v)-min(v)
        tv=sum(abs(b-a) for a,b in zip(v,v[1:]))
        windows.append(dict(start=lo,end=hi,ppp_range=ran,ppp_total_variation=tv,
            total_variation_over_range=tv/max(ran,1e-30),
            gradient_l2_median=statistics.median(x['gradient_l2'] for x in a),
            gradient_l2_max=max(x['gradient_l2'] for x in a),
            relative_update_max=max(x['relative_update_norm'] for x in a),
            obvious_oscillation=ran>=OSCILLATION['ppp_range_min'] and tv/max(ran,1e-30)>=OSCILLATION['total_variation_over_range_min']))
    conflicts=[]
    for left,right in zip(h,h[1:]):
        dl=right['total_loss']-left['total_loss']; dp=right['ppp_objective']-left['ppp_objective']
        if dl < -1e-8 and dp > 1e-8:
            conflicts.append(dict(start=left['step'],end=right['step'],delta_loss=dl,delta_ppp=dp))
    checkpoint_changes=[]
    chosen=[next(o for o in obs if o['step']==step) for step in CHECKPOINTS]
    for a,b in zip(chosen,chosen[1:]):
        checkpoint_changes.append(dict(start=a['step'],end=b['step'],
            delta_loss=b['total_loss']-a['total_loss'],delta_ppp=b['ppp_objective']-a['ppp_objective'],
            delta_intensity_correlation=b['intensity_correlation']-a['intensity_correlation'],
            delta_g_rmse=b['g_rmse']-a['g_rmse'],delta_h_std=b['h_std']-a['h_std']))
    collapse_context=[]
    if collapses:
        s=collapses[0]
        onset=min(s,near[0]) if near else s
        previous=next(o for o in obs if o['step']==onset-25)
        current=next(o for o in obs if o['step']==s)
        # Gradient at t causes update at t+1. Include the preceding gradient.
        local=[x for x in h if onset-26<=x['step']<=s]
        collapse_context.append(dict(first_observed_step=s,first_near_collapse_observed=onset,
            window_start=onset-26,window_end=s,previous_observation=previous,current_observation=current,
            max_gradient_l2=max(x['gradient_l2'] for x in local),max_relative_update=max(x['relative_update_norm'] for x in local)))
    best_obj=min(h,key=lambda x:x['ppp_objective'])
    best_corr=max(obs,key=lambda x:x['intensity_correlation'])
    best_rmse=min(obs,key=lambda x:x['intensity_rmse'])
    return dict(learning_rate=case['learning_rate'],data_seed=case['data_seed'],
        collapse=bool(collapses),branch_ever_grew=any(o['h_std']>1e-4 for o in obs),
        final_h_collapse=final['h_collapse'],final_h_near_collapse=final['h_near_collapse'],
        first_collapse_observed=collapses[0] if collapses else None,
        first_near_collapse_after_growth=near[0] if near else None,
        collapse_observed_steps=collapses,obvious_oscillation_any=any(w['obvious_oscillation'] for w in windows),
        obvious_oscillation_last200=windows[-1]['obvious_oscillation'],windows=windows,
        final=final,best_ppp_objective=best_obj['ppp_objective'],best_ppp_step=best_obj['step'],
        best_intensity_correlation=best_corr['intensity_correlation'],best_correlation_step=best_corr['step'],
        best_intensity_rmse=best_rmse['intensity_rmse'],best_rmse_step=best_rmse['step'],
        max_gradient_l2=max(x['gradient_l2'] for x in h),max_update_l2=max(x['parameter_update_l2'] for x in h),
        max_relative_update_norm=max(x['relative_update_norm'] for x in h),
        max_relative_update_post300=max(x['relative_update_norm'] for x in h if x['step']>300),
        max_gradient_l2_post300=max(x['gradient_l2'] for x in h if x['step']>300),
        loss_down_ppp_up_steps=conflicts,checkpoint_changes=checkpoint_changes,
        collapse_context=collapse_context,final_window=case['final_window'],
        baseline_replay=case['baseline_replay'])


def main():
    root=Path(__file__).resolve().parent
    report=json.loads((root/(STEM+'_RESULTS.json')).read_text())
    if report['status']!='complete': raise RuntimeError('Incomplete experiment')
    summaries=[summarize(c) for c in report['cases']]
    write_json(root/(STEM+'_SUMMARY.json'),summaries)
    cases={(c['learning_rate'],c['data_seed']):c for c in report['cases']}
    colors={.01:'#c44136',.003:'#2475b5',.001:'#258a55'}
    groups=[('OPTIMIZATION', [('ppp_objective','PPP objective','linear'),('total_loss','Total loss','linear'),
              ('gradient_l2','Gradient L2','log'),('relative_update_norm','Relative update','log')]),
            ('RECOVERY',[('intensity_correlation','Intensity correlation','linear'),('g_rmse','Predictor g RMSE','linear'),
              ('h_std','Area-weighted h std','symlog'),('spatial_contribution_std','Spatial contribution std','symlog')])]
    for name,specs in groups:
        fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,layout='constrained')
        for col,seed in enumerate(SEEDS):
            for lr in RATES:
                case=cases[(lr,seed)]
                rows=case['history'] if name=='OPTIMIZATION' else case['observations']
                x=[r['step'] for r in rows]
                for row,(key,label,scale) in enumerate(specs):
                    ax=axes[row,col]
                    y=[r[key] for r in rows]
                    if scale=='log': y=[v if v>0 else float('nan') for v in y]  # step0 has no update; raw zero remains saved
                    ax.plot(x,y,color=colors[lr],lw=.85,label=f'lr={lr}')
                    ax.set_ylabel(label); ax.grid(alpha=.2); ax.set_xlim(0,2000)
                    if scale=='log': ax.set_yscale('log')
                    if scale=='symlog': ax.set_yscale('symlog',linthresh=1e-6)
                    if key=='intensity_correlation': ax.set_ylim(-1.05,1.05)
            axes[0,col].set_title(f'data seed {seed}; init=1011; n=256')
            axes[0,col].legend(fontsize=8)
            axes[-1,col].set_xlabel('Adam step')
        fig.suptitle(f'P1-D learning-rate diagnostic: {name.lower()}\nAll other training settings fixed; no epoch selection',fontsize=14)
        fig.savefig(root/(STEM+'_'+name+'.png'),dpi=160)
        plt.close(fig)
    archive=torch.load(root/(STEM+'_STATES.pt'),weights_only=True)
    teacher=archive['teacher_g'].numpy().reshape(48,48).T
    surfaces=[teacher]+[c['checkpoints'][2000]['g_surface'].numpy().reshape(48,48).T for c in archive['cases'].values()]
    bound=max(float(np.abs(s).max()) for s in surfaces)
    t=np.linspace(-1,1,49);edges=t+.12*np.sin(np.pi*t)
    fig,axes=plt.subplots(3,4,figsize=(14,8),layout='constrained',sharex=True,sharey=True)
    for row,seed in enumerate(SEEDS):
        for col,lr in enumerate((None,)+RATES):
            surface=teacher if lr is None else archive['cases'][f'{lr}_{seed}']['checkpoints'][2000]['g_surface'].numpy().reshape(48,48).T
            ax=axes[row,col]
            mesh=ax.pcolormesh(edges*60,edges*40,surface,cmap='coolwarm',vmin=-bound,vmax=bound,shading='flat')
            ax.set_title(f'seed {seed}: '+('teacher' if lr is None else f'lr={lr}, step=2000'),fontsize=10)
            ax.set_aspect('equal')
            if row==2: ax.set_xlabel('x (km)')
            if col==0: ax.set_ylabel('y (km)')
    fig.colorbar(mesh,ax=axes,label='Raw predictor g; shared unclipped scale',shrink=.8)
    fig.suptitle('Teacher vs every final student: dense 48 x 48 predictor',fontsize=14)
    fig.savefig(root/(STEM+'_DENSE_PREDICTORS.png'),dpi=160);plt.close(fig)
    lines=['| lr | seed | collapse first observed | oscillation any / last200 | final PPP | minimum PPP @step | final intensity corr / RMSE | maximum corr @step | max gradient L2 | max relative update | final h std / spatial std |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for s in sorted(summaries,key=lambda x:(x['data_seed'],-x['learning_rate'])):
        f=s['final'];collapse=str(s['first_collapse_observed']) if s['collapse'] else 'none through 2000'
        lines.append(f"| {s['learning_rate']} | {s['data_seed']} | {collapse} | {s['obvious_oscillation_any']} / {s['obvious_oscillation_last200']} | {f['ppp_objective']:.6f} | {s['best_ppp_objective']:.6f} @{s['best_ppp_step']} | {f['intensity_correlation']:.6f} / {f['intensity_rmse']:.6f} | {s['best_intensity_correlation']:.6f} @{s['best_correlation_step']} | {s['max_gradient_l2']:.6f} | {s['max_relative_update_norm']:.6g} | {f['h_std']:.6g} / {f['spatial_contribution_std']:.6g} |")
    (root/(STEM+'_TABLE.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Saved all-run comparison plots, dense predictor maps and descriptive summary.')


if __name__=='__main__': main()
