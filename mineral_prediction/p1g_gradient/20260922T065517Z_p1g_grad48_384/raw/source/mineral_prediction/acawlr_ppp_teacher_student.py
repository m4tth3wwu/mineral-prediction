"""P1-C teacher/student: unchanged finite model, separated random effects.

Teacher recipe and all gates are fixed before student fitting. No restart selection.
Run --smoke first, then unittest with ACAWLR_P1C_OUTPUT set to an output directory.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import torch
from torch import nn
from acawlr_ppp_bridge import (ReducedRankOneSyntheticPredictor, fit_rank_one_synthetic,
                               rank_one_value_and_grad)
from acawlr_ppp_synthetic import (PENALTY, STEPS, SCALE, DOMAIN_AREA, quadrature,
                                 normalized_density, evaluate, correlation, rmse,
                                 stability_summary)

TEACHER_SEED = 20260917
DATA_SEEDS = (11, 23, 37, 53, 71)
INIT_SEEDS = (1011, 1023, 1037, 1053, 1071)
FIXED_DATA_SEED = 11
FIXED_INIT_SEED = 1011
TRAIN_RESOLUTION = 24
EVAL_RESOLUTION = 48
# Same broad recovery gates as P1-B; both experiments tested independently.
RECOVERY_GATES = dict(mean_intensity_correlation_min=.7, mean_intensity_rmse_max=.8,
                      mean_g_correlation_min=.65, mean_top20_overlap_min=.5,
                      each_intensity_correlation_min=.4)
QUADRATURE_LOGZ_TOL = .01
QUADRATURE_RELATIVE_RMSE_TOL = .03


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_teacher():
    """Fixed constructive spatial path plus seed-fixed 0.01 Gaussian perturbation.

    ASPNN starts as sigmoid(2*abs(perpendicular_offset)-2). CAWNN starts
    with four centre-copy channels, zero residual/attention logits, unit readout.
    Perturbations activate a less symmetric teacher without any student feedback.
    """
    model = ReducedRankOneSyntheticPredictor(TEACHER_SEED)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        a = model.aspnn.network
        a[0].weight[0, 1] = 1
        a[0].weight[1, 1] = -1
        a[2].weight[0, :2] = 1
        a[4].weight[0, 0] = 1
        a[6].weight[0, 0] = 2
        a[6].bias[0] = -2
        model.cawnn.head.weight[:, 0, 1, 1] = 1
        model.cawnn.readout.weight.fill_(1)
        model.beta.copy_(model.beta.new_tensor([.3, -.2]))
        model.u.copy_(model.u.new_tensor([6., 3.]))
        rng = torch.Generator().manual_seed(TEACHER_SEED)
        for module in (model.aspnn, model.cawnn):
            for p in module.parameters():
                p.add_(.01 * torch.randn(p.shape, generator=rng, dtype=p.dtype))
    return model.eval().requires_grad_(False)


def _affine_bounds(module, lo, hi):
    positive, negative = module.weight.clamp(min=0), module.weight.clamp(max=0)
    if isinstance(module, nn.Conv2d):
        def op(x, w):
            return torch.nn.functional.conv2d(x, w, padding=module.padding)
    else:
        def op(x, w):
            return torch.nn.functional.linear(x, w)
    low, high = op(lo, positive) + op(hi, negative), op(hi, positive) + op(lo, negative)
    if module.bias is not None:
        bias = module.bias[None, :, None, None] if isinstance(module, nn.Conv2d) else module.bias
        low, high = low + bias, high + bias
    return low, high


def _channel_bounds(module, lo, hi):
    lo, hi = _affine_bounds(module[0], lo, hi)
    return _affine_bounds(module[2], lo.relu(), hi.relu())


@torch.no_grad()
def rejection_bound(model):
    """Certified global score envelope, not a grid maximum.

    ASPNN sigmoid and its axial average lie in [0,1]. Interval arithmetic
    through the entire CAWNN encloses raw(s), hence h(s). |x_j|<=1 gives M.
    """
    c = model.cawnn
    lo, hi = _affine_bounds(c.head, torch.zeros((1, 1, 4, 3), dtype=torch.float64),
                            torch.ones((1, 1, 4, 3), dtype=torch.float64))
    rlo, rhi = _affine_bounds(c.residual[0], lo, hi)
    rlo, rhi = _affine_bounds(c.residual[2], rlo.relu(), rhi.relu())
    lo, hi = (lo + rlo).relu(), (hi + rhi).relu()
    alo, ahi = _channel_bounds(c.channel, lo.mean((2, 3), keepdim=True), hi.mean((2, 3), keepdim=True))
    blo, bhi = _channel_bounds(c.channel, lo.amax((2, 3), keepdim=True), hi.amax((2, 3), keepdim=True))
    lo, hi = lo * (alo + blo).sigmoid(), hi * (ahi + bhi).sigmoid()
    pooled_lo = torch.cat((lo.mean(1, keepdim=True), lo.amax(1, keepdim=True)), 1)
    pooled_hi = torch.cat((hi.mean(1, keepdim=True), hi.amax(1, keepdim=True)), 1)
    alo, ahi = _affine_bounds(c.spatial, pooled_lo, pooled_hi)
    lo, hi = lo * alo.sigmoid(), hi * ahi.sigmoid()
    lo, hi = _affine_bounds(c.readout, lo.mean((2, 3)), hi.mean((2, 3)))
    ref = model.raw_spatial(model.reference).tanh()
    h_abs = torch.maximum((lo.tanh() - ref).abs(), (hi.tanh() - ref).abs()).max()
    return float(model.beta.abs().sum() + model.u.abs().sum() * h_abs) + 1e-12


@torch.no_grad()
def sample_events(teacher, n, seed):
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    rng = torch.Generator().manual_seed(seed)
    bound = rejection_bound(teacher)
    accepted, count = [], 0
    while count < n:
        draws = torch.rand((2048, 3), generator=rng, dtype=torch.float64)
        xy = (2 * draws[:, :2] - 1) * SCALE
        scores = teacher.g(xy)
        if not bool(torch.isfinite(scores).all()) or bool((scores > bound).any()):
            raise FloatingPointError('Invalid rejection envelope or teacher score')
        keep = draws[:, 2].log() <= scores - bound
        accepted.append(xy[keep])
        count += int(keep.sum())
    return torch.cat(accepted)[:n]


@torch.no_grad()
def refinement(model):
    grid, area = quadrature(EVAL_RESOLUTION)
    scores = model.g(grid)
    values, surfaces = [], []
    for resolution in (24, 48, 96):
        q, w = quadrature(resolution)
        _, z = normalized_density(model.g(q), w)
        values.append(dict(resolution=resolution, log_z=float(z)))
        surfaces.append(DOMAIN_AREA * (scores - z).exp())
    comparisons = []
    for i, j in ((0, 1), (1, 2)):
        delta = values[j]['log_z'] - values[i]['log_z']
        error = rmse(surfaces[i], surfaces[j], area)
        comparisons.append(dict(coarse=values[i]['resolution'], fine=values[j]['resolution'],
                                delta_log_z=delta, intensity_rmse=error,
                                density_rmse=error / DOMAIN_AREA,
                                intensity_correlation=correlation(surfaces[i], surfaces[j], area),
                                stable=abs(delta) < QUADRATURE_LOGZ_TOL and error < QUADRATURE_RELATIVE_RMSE_TOL))
    return dict(values=values, comparisons=comparisons)


@torch.no_grad()
def exact_copy_sanity(teacher):
    student = ReducedRankOneSyntheticPredictor(0)
    student.load_state_dict(copy.deepcopy(teacher.state_dict()))
    q, area = quadrature(96)
    a, _ = normalized_density(student.g(q), area)
    b, _ = normalized_density(teacher.g(q), area)
    return dict(g_max_abs_error=float((student.g(q) - teacher.g(q)).abs().max()),
                coefficient_max_abs_error=float((student.local_coefficients(q) - teacher.local_coefficients(q)).abs().max()),
                density_max_abs_error=float((a - b).abs().max()),
                relative_intensity_max_abs_error=float(((a - b) * DOMAIN_AREA).abs().max()))


@torch.no_grad()
def teacher_diagnostics(teacher):
    q, area = quadrature(96)
    g, h = teacher.g(q), teacher.h(q)
    p, _ = normalized_density(g, area)
    def stats(x):
        mean = (x * area).sum() / area.sum()
        return dict(min=float(x.min()), max=float(x.max()), mean=float(mean),
                    std=float(((x - mean).square() * area / area.sum()).sum().sqrt()))
    order = torch.argsort(p, descending=True)
    # Fractional final cell gives exactly 20% area for this concentration statistic.
    used = (.2 * area.sum() - (area[order].cumsum(0) - area[order])).clamp(min=0)
    used = torch.minimum(used, area[order])
    metrics, _ = evaluate(teacher, teacher, q, area)
    return dict(seed=TEACHER_SEED, state_sha256=state_hash(teacher.state_dict()),
                parameters=sum(p.numel() for p in teacher.parameters()),
                beta=teacher.beta.tolist(), u=teacher.u.tolist(),
                parameter_norms={k: float(v.norm()) for k, v in teacher.named_parameters()},
                g=stats(g), h=stats(h), spatial_score=stats(g - teacher.features(q) @ teacher.beta),
                intensity_concentration=dict(max_relative=float(p.max() * DOMAIN_AREA),
                    min_relative=float(p.min() * DOMAIN_AREA), top20_mass=float((p[order] * used).sum()),
                    relative_second_moment=float(((p * DOMAIN_AREA).square() * area / area.sum()).sum())),
                all_finite=all(bool(torch.isfinite(x).all()) for x in (g, h, p, teacher.local_coefficients(q))),
                reference_error=metrics['reference_error'], rank_one_residual=metrics['rank_one_residual'],
                penalty=float(PENALTY(teacher)), rejection_score_bound=rejection_bound(teacher),
                refinement=refinement(teacher), exact_copy=exact_copy_sanity(teacher))


def recovery_failures(cases, summary):
    failures = []
    for metric, direction, threshold in (
        ('intensity_correlation', 'min', .7), ('intensity_rmse', 'max', .8),
        ('g_correlation', 'min', .65), ('top20_overlap', 'min', .5)):
        value = summary[metric]['mean']
        if not (value > threshold if direction == 'min' else value < threshold):
            failures.append(f'mean {metric}={value:.8g} fails {direction} {threshold}')
    for case in cases:
        if case['intensity_correlation'] <= .4:
            failures.append(f"{case['id']}: intensity correlation <= 0.4")
    return failures


def fit_student(events, initial_state, *, steps=STEPS):
    """No teacher argument or surface is supplied to the training interface."""
    model = ReducedRankOneSyntheticPredictor(0)
    model.load_state_dict(copy.deepcopy(initial_state))
    q, area = quadrature(TRAIN_RESOLUTION)
    history = fit_rank_one_synthetic(model, events, q, area, penalty=PENALTY,
                                    steps=steps, learning_rate=.01)
    _, _, _, grads = rank_one_value_and_grad(model, events, q, area, PENALTY)
    return model, history, dict(finite=all(bool(torch.isfinite(g).all()) for g in grads),
                               l2=float(sum(g.square().sum() for g in grads).sqrt()),
                               max_abs=max(float(g.abs().max()) for g in grads))


def prepare_design(teacher):
    data = {seed: sample_events(teacher, 256, seed) for seed in DATA_SEEDS}
    states = {seed: copy.deepcopy(ReducedRankOneSyntheticPredictor(seed).state_dict()) for seed in INIT_SEEDS}
    cases = []
    for experiment in ('A', 'B'):
        for n in (44, 256):
            for seed in (INIT_SEEDS if experiment == 'A' else DATA_SEEDS):
                ds = FIXED_DATA_SEED if experiment == 'A' else seed
                ins = seed if experiment == 'A' else FIXED_INIT_SEED
                cases.append(dict(id=f'{experiment}_n{n}_data{ds}_init{ins}', experiment=experiment,
                                  n=n, data_seed=ds, init_seed=ins, events=data[ds][:n].clone(),
                                  initial_state=copy.deepcopy(states[ins])))
    return cases


def run_experiments(*, smoke=False):
    teacher = make_teacher()
    diagnostics = teacher_diagnostics(teacher)
    # Teacher-only gates: fail before any recovery observation, never replace teacher.
    assert diagnostics['all_finite']
    assert diagnostics['h']['std'] > .02
    assert diagnostics['spatial_score']['std'] > .1
    assert .15 < diagnostics['g']['std'] < 2
    assert diagnostics['intensity_concentration']['max_relative'] < 10
    assert diagnostics['refinement']['comparisons'][0]['stable']
    assert max(diagnostics['exact_copy'].values()) < 1e-12
    report = dict(stage='P1-C Teacher-Student', teacher=diagnostics,
                  config=dict(teacher_seed=TEACHER_SEED, data_seeds=DATA_SEEDS, init_seeds=INIT_SEEDS,
                              fixed_data_seed=FIXED_DATA_SEED, fixed_init_seed=FIXED_INIT_SEED,
                              steps=2 if smoke else STEPS, learning_rate=.01, penalty=vars(PENALTY),
                              train_resolution=24, evaluation_resolution=48, refinement_resolutions=[24,48,96],
                              restarts=1, dtype='float64', torch_version=torch.__version__, threads=1,
                              domain_area_km2=DOMAIN_AREA, std_ddof=1,
                              intensity_rmse_unit='A*p (area mean 1)', density_rmse_unit='km^-2',
                              recovery_gates=RECOVERY_GATES, quadrature_logz_tolerance=QUADRATURE_LOGZ_TOL,
                              quadrature_relative_rmse_tolerance=QUADRATURE_RELATIVE_RMSE_TOL),
                  runs=[], summaries={})
    archive = dict(teacher_state=copy.deepcopy(teacher.state_dict()), cases={})
    design = prepare_design(teacher)
    if smoke:
        design = [design[i] for i in (0, 1, 5, 6, 10, 11, 15, 16)]
    grid, area = quadrature(48)
    for spec in design:
        events, initial = spec['events'], spec['initial_state']
        model, history, gradient = fit_student(events, initial, steps=2 if smoke else STEPS)
        metrics, surface = evaluate(model, teacher, grid, area)
        metrics.update({k: v for k, v in spec.items() if k not in ('events', 'initial_state')})
        metrics.update(event_sha256=state_hash({'events': events}), initial_state_sha256=state_hash(initial),
                       initial=history[0], final=history[-1], final_gradient=gradient,
                       quadrature=refinement(model))
        with torch.no_grad():
            h = model.h(grid)
            coefficients = model.local_coefficients(grid)
            singular = torch.linalg.svdvals(coefficients - model.beta)
            metrics['rank_one_absolute_residual'] = float(singular[1])
            metrics['rank_one_roundoff_bound'] = float(64 * torch.finfo(coefficients.dtype).eps * coefficients.norm())
            mean = (h * area / area.sum()).sum()
            metrics['h_std'] = float(((h-mean).square() * area / area.sum()).sum().sqrt())
            metrics['spatial_score_rms'] = rmse(model.g(grid), model.features(grid) @ model.beta, area)
            metrics['teacher_profile_objective_on_same_events'] = float(
                len(events) * normalized_density(teacher.g(quadrature(24)[0]), quadrature(24)[1])[1] - teacher.g(events).sum())
            metrics['teacher_penalized_objective_on_same_events'] = metrics['teacher_profile_objective_on_same_events'] + diagnostics['penalty']
        report['runs'].append(metrics)
        archive['cases'][spec['id']] = dict(events=events, initial_state=initial,
                    final_state=copy.deepcopy(model.state_dict()), history=history, relative_intensity=surface)
        print(f"P1-C {spec['id']}: corr={metrics['intensity_correlation']:.6f} "
              f"RMSE={metrics['intensity_rmse']:.6f} h_std={metrics['h_std']:.6g}", flush=True)
    for experiment in ('A', 'B'):
        report['summaries'][experiment] = {}
        for n in (44, 256):
            cases = [c for c in report['runs'] if c['experiment'] == experiment and c['n'] == n]
            surfaces = [archive['cases'][c['id']]['relative_intensity'] for c in cases]
            summary = stability_summary(cases, surfaces, area)
            pairs = []
            for i, j in itertools.combinations(range(len(cases)), 2):
                pairs.append(dict(left=cases[i]['id'], right=cases[j]['id'],
                                  correlation=correlation(surfaces[i], surfaces[j], area),
                                  rmse=rmse(surfaces[i], surfaces[j], area)))
            summary['pairwise'] = pairs
            for key in ('correlation', 'rmse'):
                values = torch.tensor([p[key] for p in pairs], dtype=torch.float64)
                summary['pairwise_' + key] = dict(mean=float(values.mean()),
                    std=float(values.std(correction=1)) if len(values)>1 else None,
                    min=float(values.min()), max=float(values.max()))
            summary['recovery_failures'] = recovery_failures(cases, summary)
            report['summaries'][experiment][str(n)] = summary
    # Extremes describe every saved result; never select a model for deployment.
    ordered = sorted(report['runs'], key=lambda c: c['intensity_correlation'])
    report['quadrature_diagnostic_examples'] = dict(lowest_correlation=ordered[0]['id'],
                                                   highest_correlation=ordered[-1]['id'])
    return report, archive


def save_report(report, archive, directory, stem='ACAWLR_PPP_P1C_TS'):
    directory = Path(directory)
    torch.save(archive, directory / (stem + '_STATES.pt'))
    report['archive'] = stem + '_STATES.pt'
    (directory / (stem + '_RESULTS.json')).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report, archive = run_experiments(smoke=args.smoke)
    save_report(report, archive, args.output, 'ACAWLR_PPP_P1C_TS_SMOKE' if args.smoke else 'ACAWLR_PPP_P1C_TS')
