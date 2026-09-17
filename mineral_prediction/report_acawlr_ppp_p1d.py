"""Read saved P1-D trajectories; render every recorded point, no fit or selection."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acawlr_ppp_optimization import STEM, CHECKPOINTS, write_json


def main():
    root=Path(__file__).resolve().parent
    report=json.loads((root/(STEM+'_RESULTS.json')).read_text())
    if report['status']!='complete':
        raise RuntimeError('Run incomplete; do not render a final diagnostic report')
    fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,layout='constrained')
    summaries=[]
    for col,case in enumerate(report['cases']):
        history,obs=case['history'],case['observations']
        x=[r['step'] for r in history]
        xo=[r['step'] for r in obs]
        axes[0,col].plot(x,[r['objective'] for r in history],lw=.8)
        axes[1,col].plot(x,[max(r['gradient_l2_per_n'],1e-16) for r in history],lw=.7)
        axes[1,col].set_yscale('log')
        axes[2,col].plot(xo,[r['intensity_correlation'] for r in obs],lw=1.1)
        axes[2,col].set_ylim(-1.05,1.05)
        axes[3,col].plot(xo,[r['h_std'] for r in obs],lw=1.1)
        axes[3,col].set_yscale('symlog',linthresh=1e-6)
        for row,label in enumerate(('Total penalized objective','Gradient L2 / n (log)','Intensity correlation','Area-weighted h std (symlog)')):
            ax=axes[row,col]
            ax.set_ylabel(label)
            ax.axvline(300,color='black',ls='--',lw=.8)
            ax.grid(alpha=.2)
            ax.set_xlim(0,2000)
        axes[0,col].set_title(f"data seed {case['data_seed']}, init 1011, n=256")
        axes[3,col].set_xlabel('Adam step')
        blocks=[]
        for lo,hi in ((301,600),(601,1000),(1001,1500),(1501,2000),(1801,2000)):
            segment=[r for r in history if lo<=r['step']<=hi]
            grads=[r['gradient_l2_per_n'] for r in segment]
            objectives=[r['objective'] for r in segment]
            blocks.append(dict(first=lo,last=hi,objective_min=min(objectives),objective_max=max(objectives),
                gradient_per_n_min=min(grads),gradient_per_n_median=statistics.median(grads),gradient_per_n_max=max(grads),
                gradient_explosion_steps=sum(r['gradient_explosion'] for r in segment)))
        chosen=[next(o for o in obs if o['step']==s) for s in (300,600,1000,1500,2000)]
        differences=[]
        for left,right in zip(chosen,chosen[1:]):
            differences.append(dict(start=left['step'],end=right['step'],
                delta_objective=right['objective']-left['objective'],
                delta_intensity_correlation=right['intensity_correlation']-left['intensity_correlation'],
                delta_intensity_rmse=right['intensity_rmse']-left['intensity_rmse'],
                delta_g_correlation=right['g_correlation']-left['g_correlation']))
        summaries.append(dict(data_seed=case['data_seed'],blocks=blocks,fixed_interval_differences=differences,
            final_window=case['final_window'],
            gradient_explosion_steps=[r['step'] for r in history if r['gradient_explosion']],
            nan_inf_steps=[r['step'] for r in history if r['nan_inf']],
            h_collapse_observed_steps=[r['step'] for r in obs if r['h_collapse']],
            h_near_collapse_observed_steps=[r['step'] for r in obs if r['h_near_collapse']],
            sharp_spike_observed_steps=[r['step'] for r in obs if r['sharp_intensity_spike']],
            max_gradient_l2_per_n=max(r['gradient_l2_per_n'] for r in history),
            max_peak_relative_intensity=max(r['peak_relative_intensity'] for r in obs)))
    fig.suptitle('P1-D: same archived inputs and Adam settings, extended horizon only\nDashed line: reproduced step300; no best-epoch selection',fontsize=14)
    fig.savefig(root/(STEM+'_TRAJECTORIES.png'),dpi=160)
    plt.close(fig)
    write_json(root/(STEM+'_TRAJECTORY_SUMMARY.json'),summaries)
    lines=['| data seed | step | profile objective | penalty | total objective | gradient L2 | gradient/n | intensity corr | intensity RMSE | g corr | g RMSE | coefficient RMSE | top20 | h std | u norm | spatial std | logZ | b | peak A*p |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    keys=('profile_objective','penalty','objective','gradient_l2','gradient_l2_per_n','intensity_correlation','intensity_rmse','g_correlation','g_rmse','coefficient_rmse','top20_overlap','h_std','u_norm','spatial_contribution_std','log_z','intercept','peak_relative_intensity')
    for c in report['cases']:
        for step in CHECKPOINTS:
            o=next(o for o in c['observations'] if o['step']==step)
            lines.append('| '+ ' | '.join([str(c['data_seed']),str(step)]+[f'{o[k]:.9g}' for k in keys])+' |')
    (root/(STEM+'_CHECKPOINT_TABLE.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Rendered trajectories, all checkpoint metrics, and descriptive summaries.')


if __name__=='__main__':
    main()
