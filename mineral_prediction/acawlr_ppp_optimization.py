"""P1-D: three archived n=256 inputs, one uninterrupted Adam trajectory each.

No sampling, new initial conditions, truth-based selection or early stopping.
Every case must reproduce P1-C at step300 before any case passes that barrier.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
import torch
from acawlr_ppp_bridge import ReducedRankOneSyntheticPredictor, rank_one_value_and_grad
from acawlr_ppp_synthetic import PENALTY, quadrature, evaluate
from acawlr_ppp_teacher_student import state_hash

SEEDS = (23, 71, 53)
CHECKPOINTS = (0, 100, 300, 600, 1000, 1500, 2000)
HORIZON = 2000
EVALUATE_EVERY = 25
STEM = 'ACAWLR_PPP_P1D_OPTIMIZATION'
# Descriptive diagnostics, fixed before running; none affect updates or stopping.
DIAGNOSTICS = dict(h_collapse_max=1e-12, h_near_collapse_max=1e-4,
                   gradient_explosion_per_event_min=10., sharp_peak_relative_min=10.,
                   convergence_window=200, convergence_max_gradient_per_event=1e-3,
                   convergence_objective_range_per_event=1e-5)
REPRODUCTION = dict(rtol=1e-9, atol=1e-10, history_max_abs_tolerance=1e-8)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_manifest(root):
    repo = root.parent
    tracked = subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    paths = [p for p in tracked if ('acawlr_ppp' in Path(p).name.lower() and 'p1d' not in p.lower())
             or p == 'scripts/ACAWLR_improved.py']
    return {p: sha256(repo/p) for p in paths}


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def load_inputs(root):
    report = json.loads((root/'ACAWLR_PPP_P1C_TS_RESULTS.json').read_text(encoding='utf-8'))
    archive = torch.load(root/'ACAWLR_PPP_P1C_TS_STATES.pt',weights_only=True)
    teacher = ReducedRankOneSyntheticPredictor(0)
    teacher.load_state_dict(archive['teacher_state'])
    teacher.requires_grad_(False).eval()
    assert state_hash(teacher.state_dict()) == report['teacher']['state_sha256']
    assert report['config']['torch_version'] == torch.__version__
    assert report['config']['penalty'] == vars(PENALTY) == dict(beta=.02,u=.2,theta=.001)
    assert report['config']['train_resolution'] == 24
    assert report['config']['evaluation_resolution'] == 48
    assert report['config']['learning_rate'] == .01
    inputs = []
    for seed in SEEDS:
        key = f'B_n256_data{seed}_init1011'
        saved = archive['cases'][key]
        old = next(c for c in report['runs'] if c['id']==key)
        assert saved['events'].shape == (256,2)
        assert state_hash({'events':saved['events']}) == old['event_sha256']
        assert state_hash(saved['initial_state']) == old['initial_state_sha256']
        inputs.append((seed,saved,old))
    return teacher, inputs


class Trajectory:
    """Optimizer sees only archived events, fixed quadrature and existing penalty."""
    def __init__(self, seed, saved):
        self.seed = seed
        self.events = saved['events'].clone()
        self.initial_state = copy.deepcopy(saved['initial_state'])
        self.model = ReducedRankOneSyntheticPredictor(0)
        self.model.load_state_dict(self.initial_state)
        self.model.train()
        self.q, self.area = quadrature(24)
        self.grid, self.weights = quadrature(48)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=.01)
        self.step = 0
        self.history = []
        self.observations = []
        self.checkpoints = {}
        self.gradients = None
        self.reproduction = None
        self.error = None

    def observe(self, teacher):
        result, penalty, total, self.gradients = rank_one_value_and_grad(
            self.model,self.events,self.q,self.area,PENALTY)
        norm = float(sum(g.square().sum() for g in self.gradients).sqrt())
        row = dict(step=self.step, profile_objective=float(result.objective.detach()),
                   penalty=float(penalty), objective=float(total), gradient_l2=norm,
                   gradient_l2_per_n=norm/256, gradient_max_abs=max(float(g.abs().max()) for g in self.gradients),
                   log_z=float(result.log_z.detach()), intercept=float(result.intercept.detach()),
                   nan_inf=False, gradient_explosion=norm/256>DIAGNOSTICS['gradient_explosion_per_event_min'])
        self.history.append(row)
        if self.step % EVALUATE_EVERY == 0 or self.step in CHECKPOINTS:
            with torch.no_grad():
                metrics,surface = evaluate(self.model,teacher,self.grid,self.weights)
                h = self.model.h(self.grid)
                # Exact same arithmetic ordering for the P1-C h std comparison.
                mean = (h*self.weights/self.weights.sum()).sum()
                h_std = float(((h-mean).square()*self.weights/self.weights.sum()).sum().sqrt())
                spatial = (self.model.features(self.grid)*self.model.u).sum(1)*h
                spatial_mean = (spatial*self.weights/self.weights.sum()).sum()
                spatial_std = float(((spatial-spatial_mean).square()*self.weights/self.weights.sum()).sum().sqrt())
                metrics.update(row)
                metrics.update(h_std=h_std,u_norm=float(self.model.u.norm()),spatial_contribution_std=spatial_std,
                               peak_relative_intensity=float(surface.max()),
                               h_collapse=h_std<=DIAGNOSTICS['h_collapse_max'],
                               h_near_collapse=h_std<=DIAGNOSTICS['h_near_collapse_max'],
                               sharp_intensity_spike=float(surface.max())>DIAGNOSTICS['sharp_peak_relative_min'])
                if not all(math.isfinite(v) for v in metrics.values() if isinstance(v,float)):
                    raise FloatingPointError('Nonfinite evaluation metric')
            self.observations.append(metrics)
        if self.step in CHECKPOINTS:
            self.checkpoints[self.step] = dict(model_state=copy.deepcopy(self.model.state_dict()),
                optimizer_state=copy.deepcopy(self.optimizer.state_dict()),
                relative_intensity=surface.clone())
            print(f'P1-D data={self.seed} step={self.step}: objective={row["objective"]:.9f} '
                  f'grad/n={norm/256:.6g} corr={metrics["intensity_correlation"]:.6f} '
                  f'h_std={metrics["h_std"]:.6g}',flush=True)

    def advance(self, target, teacher):
        if not self.history:
            self.observe(teacher)
        while self.step < target:
            self.optimizer.zero_grad(set_to_none=True)
            for parameter,gradient in zip(self.model.parameters(),self.gradients):
                parameter.grad = gradient
            self.optimizer.step()
            self.step += 1
            self.observe(teacher)

    def compare_step300(self, old, saved):
        current = next(c for c in self.observations if c['step']==300)
        pairs = dict(objective=old['final']['objective'],
                     profile_objective=old['final']['profile_objective'], penalty=old['final']['penalty'],
                     gradient_l2=old['final_gradient']['l2'], h_std=old['h_std'],
                     **{k:old[k] for k in ('intensity_correlation','intensity_rmse','g_correlation','g_rmse')})
        checks = {k:dict(previous=v,current=current[k],abs_error=abs(v-current[k]),
                        passed=math.isclose(v,current[k],rel_tol=REPRODUCTION['rtol'],abs_tol=REPRODUCTION['atol']))
                  for k,v in pairs.items()}
        model_close = all(torch.allclose(v,saved['final_state'][k],rtol=REPRODUCTION['rtol'],atol=REPRODUCTION['atol'])
                          for k,v in self.model.state_dict().items())
        history_error = max(abs(a[k]-b[k]) for a,b in zip(self.history,saved['history'])
                            for k in ('objective','profile_objective','penalty','log_z','intercept'))
        self.reproduction = dict(metrics=checks, model_state_close=model_close,
            model_state_bit_identical=all(torch.equal(v,saved['final_state'][k]) for k,v in self.model.state_dict().items()),
            history_max_abs_error=history_error,
            passed=all(c['passed'] for c in checks.values()) and model_close and history_error<=REPRODUCTION['history_max_abs_tolerance'])
        return self.reproduction['passed']

    def as_report(self):
        tail = self.history[-DIAGNOSTICS['convergence_window']:]
        grad_max = max(r['gradient_l2_per_n'] for r in tail)
        objective_range = (max(r['objective'] for r in tail)-min(r['objective'] for r in tail))/256
        return dict(data_seed=self.seed,init_seed=1011,n=256,last_step=self.step,
                    event_sha256=state_hash({'events':self.events}),initial_state_sha256=state_hash(self.initial_state),
                    reproduction=self.reproduction,error=self.error,history=self.history,observations=self.observations,
                    final_window=dict(first_step=tail[0]['step'],last_step=tail[-1]['step'],
                        max_gradient_l2_per_n=grad_max,objective_range_per_n=objective_range,
                        converged=len(tail)==DIAGNOSTICS['convergence_window'] and grad_max<=DIAGNOSTICS['convergence_max_gradient_per_event']
                            and objective_range<=DIAGNOSTICS['convergence_objective_range_per_event']))


def run(root):
    torch.set_num_threads(1)
    teacher, inputs = load_inputs(root)
    trajectories = [Trajectory(seed,saved) for seed,saved,_ in inputs]
    baseline = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip()
    manifest = history_manifest(root)
    protocol = dict(baseline_commit=baseline,data_seeds=SEEDS,init_seed=1011,n=256,horizon=HORIZON,
                    checkpoint_steps=CHECKPOINTS,evaluate_every=EVALUATE_EVERY,early_stopping=False,
                    penalty=vars(PENALTY),training_quadrature=24,evaluation_quadrature=48,
                    optimizer='torch.optim.Adam',optimizer_defaults=trajectories[0].optimizer.defaults,
                    torch_version=torch.__version__,dtype='float64',threads=1,
                    reproduction_tolerances=REPRODUCTION,diagnostic_thresholds=DIAGNOSTICS,
                    teacher_sha256=state_hash(teacher.state_dict()),historical_sha256=manifest,
                    source_sha256=sha256(Path(__file__)))
    write_json(root/(STEM+'_PROTOCOL.json'),protocol)
    status='running_step300_barrier'
    start=time.perf_counter()
    def save():
        write_json(root/(STEM+'_RESULTS.json'),dict(protocol=protocol,status=status,seconds=time.perf_counter()-start,
                    historical_files_unchanged=history_manifest(root)==manifest,
                    cases=[t.as_report() for t in trajectories if t.history]))
        torch.save(dict(teacher_state=copy.deepcopy(teacher.state_dict()),
                        cases={t.seed:dict(events=t.events,initial_state=t.initial_state,checkpoints=t.checkpoints)
                               for t in trajectories}),root/(STEM+'_STATES.pt'))
    try:
        # Runtime guards make forbidden losses and resampling fail immediately.
        with patch('torch.nn.functional.binary_cross_entropy',side_effect=AssertionError('BCE forbidden')), \
             patch('torch.nn.functional.binary_cross_entropy_with_logits',side_effect=AssertionError('BCE forbidden')), \
             patch('acawlr_ppp_teacher_student.sample_events',side_effect=AssertionError('Resampling forbidden')):
            for trajectory,(_,saved,old) in zip(trajectories,inputs):
                trajectory.advance(300,teacher)
                passed=trajectory.compare_step300(old,saved)
                save()
                if not passed:
                    status='stopped_reproduction_failure'
                    save()
                    raise RuntimeError(f'Step300 reproduction failed for seed {trajectory.seed}; no long-run interpretation allowed')
            status='running_long_horizon'
            print('All three step300 reproduction gates passed; keeping all optimizer states intact.',flush=True)
            for trajectory in trajectories:
                for target in (600,1000,1500,2000):
                    trajectory.advance(target,teacher)
                    save()
            status='complete'
            save()
    except Exception as exc:
        if status!='stopped_reproduction_failure':
            status='stopped_error'
            trajectory.error=dict(type=type(exc).__name__,message=str(exc),step=trajectory.step,
                                  nan_inf=isinstance(exc,FloatingPointError))
            save()
        raise
    return root/(STEM+'_RESULTS.json')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    if (args.output/(STEM+'_RESULTS.json')).exists():
        raise SystemExit('Refusing to replace an existing P1-D run; preserve it before a deliberate rerun.')
    print(run(args.output))
