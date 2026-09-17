"""P1-D LR-only diagnostic, extending the frozen Trajectory without replacing it.

Legacy internal `objective` is total loss; exported fields explicitly distinguish
ppp_objective from total_loss. Baseline replays only fill missing update diagnostics.
"""
import copy
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
import torch
from acawlr_ppp_optimization import (Trajectory, load_inputs, CHECKPOINTS, HORIZON,
    EVALUATE_EVERY, DIAGNOSTICS, sha256, write_json)
from acawlr_ppp_synthetic import PENALTY
from acawlr_ppp_teacher_student import state_hash

STEM='ACAWLR_PPP_P1D_LR'
RATES=(.01,.003,.001)
SEEDS=(23,53,71)
# Descriptive oscillation rule, fixed before observing the new runs.
OSCILLATION=dict(ppp_range_min=1., total_variation_over_range_min=5.,
                 windows=((301,600),(601,1000),(1001,1500),(1501,2000),(1801,2000)))


def frozen_manifest(root):
    files=subprocess.check_output(['git','ls-files'],cwd=root.parent,text=True).splitlines()
    return {p:sha256(root.parent/p) for p in files
            if ('acawlr_ppp' in Path(p).name.lower() and 'p1d_lr' not in p.lower())
            or p=='scripts/ACAWLR_improved.py'}


def vector(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def exported(row):
    out=copy.deepcopy(row)
    out['total_loss']=out.pop('objective')
    out['ppp_objective']=out.pop('profile_objective')
    out['regularization_total']=out.pop('penalty')
    return out


class LearningRateTrajectory(Trajectory):
    def __init__(self,seed,saved,lr):
        super().__init__(seed,saved)
        # No updates/moments exist yet. This is the sole changed optimizer setting.
        self.lr=lr
        self.optimizer.defaults['lr']=lr
        self.optimizer.param_groups[0]['lr']=lr
        self.previous_vector=None
        self.baseline_replay=None

    def observe(self,teacher):
        super().observe(teacher)
        current=vector(self.model)
        previous=current if self.previous_vector is None else self.previous_vector
        update=current-previous
        update_l2=float(update.norm())
        previous_l2=float(previous.norm())
        with torch.no_grad():
            parts=dict(beta=float(PENALTY.beta*self.model.beta.square().sum()/2),
                       u=float(PENALTY.u*self.model.u.square().sum()/2),
                       theta=float(PENALTY.theta*sum(p.square().sum() for m in (self.model.aspnn,self.model.cawnn)
                                                    for p in m.parameters())/2))
        extra=dict(parameter_update_l2=update_l2,relative_update_norm=update_l2/max(previous_l2,1e-30),
                   parameter_l2=float(current.norm()),previous_parameter_l2=previous_l2,
                   regularization_components=parts)
        self.history[-1].update(extra)
        if self.observations and self.observations[-1]['step']==self.step:
            self.observations[-1].update(extra)
            with torch.no_grad():
                h=self.model.h(self.grid)
                self.observations[-1]['h_range']=float(h.max()-h.min())
        if self.step in CHECKPOINTS:
            cp=self.checkpoints[self.step]
            cp['previous_parameter_vector']=previous.clone()
            cp['parameter_vector']=current.clone()
            with torch.no_grad():
                cp['g_surface']=self.model.g(self.grid).clone()
                cp['coefficient_surface']=self.model.local_coefficients(self.grid).clone()
                cp['h_surface']=self.model.h(self.grid).clone()
            print(f'LR={self.lr} seed={self.seed} step={self.step}: '
                  f'PPP={self.history[-1]["profile_objective"]:.9f}, '
                  f'loss={self.history[-1]["objective"]:.9f}, relative_update={extra["relative_update_norm"]:.6g}',flush=True)
        self.previous_vector=current

    def report(self):
        data=super().as_report()
        data['learning_rate']=self.lr
        data['baseline_replay']=self.baseline_replay
        data['history']=[exported(x) for x in self.history]
        data['observations']=[exported(x) for x in self.observations]
        data['final_window']['total_loss_range_per_n']=data['final_window'].pop('objective_range_per_n')
        return data

    def verify_baseline(self,old,states):
        fields=('objective','profile_objective','penalty','gradient_l2','gradient_l2_per_n','log_z','intercept')
        error=max(abs(a[k]-b[k]) for a,b in zip(self.history,old['history']) for k in fields)
        obs_keys=('intensity_correlation','intensity_rmse','g_correlation','g_rmse','h_std','spatial_contribution_std')
        observation_error=max(abs(a[k]-b[k]) for a,b in zip(self.observations,old['observations']) for k in obs_keys)
        def equal(a,b):
            if isinstance(a,torch.Tensor): return torch.equal(a,b)
            if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        exact=all(equal(self.checkpoints[s]['model_state'],states['checkpoints'][s]['model_state']) and
                  equal(self.checkpoints[s]['optimizer_state'],states['checkpoints'][s]['optimizer_state']) for s in CHECKPOINTS)
        self.baseline_replay=dict(history_max_abs_error=error,observation_max_abs_error=observation_error,
                                  all_model_optimizer_checkpoints_bit_identical=exact,
                                  passed=error==0 and observation_error==0 and exact)
        return self.baseline_replay['passed']


def run(root):
    torch.set_num_threads(1)
    teacher,inputs=load_inputs(root)
    saved={seed:state for seed,state,_ in inputs}
    old=json.loads((root/'ACAWLR_PPP_P1D_OPTIMIZATION_RESULTS.json').read_text())
    old_states=torch.load(root/'ACAWLR_PPP_P1D_OPTIMIZATION_STATES.pt',weights_only=True)
    manifest=frozen_manifest(root)
    protocol=dict(baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip(),
                  learning_rates=RATES,seeds=SEEDS,init_seed=1011,n=256,horizon=HORIZON,
                  checkpoints=CHECKPOINTS,evaluate_every=EVALUATE_EVERY,early_stopping=False,
                  unchanged_baseline_protocol=old['protocol'],diagnostics=DIAGNOSTICS,oscillation=OSCILLATION,
                  update_definition='row t measures theta_t-theta_(t-1); gradient at row t generates update at t+1',
                  relative_update_definition='L2(actual update) / max(L2(previous full trainable parameter vector),1e-30)',
                  objective_definition='ppp_objective=n*logZ-sum_event(g); total_loss=ppp_objective+regularization_total',
                  baseline_replay_reason='Original archive lacks per-step parameter updates; replay only supplements these diagnostics and must exactly reproduce all original checkpoints/history.',
                  historical_sha256=manifest,runner_sha256=sha256(__file__))
    write_json(root/(STEM+'_PROTOCOL.json'),protocol)
    done=[]; archives={}; current=None; status='running'; start=time.perf_counter()
    def save():
        reports=done+([current.report()] if current is not None else [])
        # Compact JSON preserves every raw row while keeping the repository manageable.
        (root/(STEM+'_RESULTS.json')).write_text(json.dumps(dict(protocol=protocol,status=status,
            seconds=time.perf_counter()-start,cases=reports,historical_files_unchanged=frozen_manifest(root)==manifest),
            ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
        if current is not None:
            archives[f'{current.lr}_{current.seed}']=dict(events=current.events,initial_state=current.initial_state,
                checkpoints=current.checkpoints)
        with torch.no_grad():
            torch.save(dict(teacher_state=teacher.state_dict(),teacher_g=teacher.g(trajectories_grid),
                teacher_coefficients=teacher.local_coefficients(trajectories_grid),cases=archives),root/(STEM+'_STATES.pt'))
    from acawlr_ppp_synthetic import quadrature
    trajectories_grid,_=quadrature(48)
    try:
        with patch('torch.nn.functional.binary_cross_entropy',side_effect=AssertionError('BCE forbidden')), \
             patch('torch.nn.functional.binary_cross_entropy_with_logits',side_effect=AssertionError('BCE forbidden')), \
             patch('acawlr_ppp_teacher_student.sample_events',side_effect=AssertionError('Resampling forbidden')):
            for lr in RATES:
                for seed in SEEDS:
                    current=LearningRateTrajectory(seed,saved[seed],lr)
                    print(f'START lr={lr} seed={seed}; inherited log objective label means total_loss.',flush=True)
                    for step in CHECKPOINTS[1:]:
                        current.advance(step,teacher)
                        save()
                    if lr==.01:
                        baseline=next(c for c in old['cases'] if c['data_seed']==seed)
                        if not current.verify_baseline(baseline,old_states['cases'][seed]):
                            raise RuntimeError('Instrumented baseline replay differs from original; halt comparison')
                    done.append(current.report()); current=None
                    save()
            status='complete'; save()
    except Exception as exc:
        status='stopped_error'
        if current is not None:
            current.error=dict(type=type(exc).__name__,message=str(exc),step=current.step)
        save()
        raise


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    if (root/(STEM+'_RESULTS.json')).exists():
        raise SystemExit('Refusing to overwrite existing LR diagnostic results')
    run(root)
