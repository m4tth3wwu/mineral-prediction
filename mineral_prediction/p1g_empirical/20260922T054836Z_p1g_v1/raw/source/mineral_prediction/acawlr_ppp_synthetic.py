"""P1-B synthetic experiment; analytic truth and continuous conditional PPP draws.

No real-data readers. Domain is [-60,60] x [-40,40] km. Fitting uses the
unchanged P1-A predictor/covariates; analytic rank-1 truth is not a teacher
network or a claim of exact finite-network representability.
"""
import itertools
import math

import torch

from acawlr_ppp_bridge import (ReducedRankOneSyntheticPredictor, RankOnePenalty,
                               fit_rank_one_synthetic)

SEEDS = (11, 23, 37, 53, 71)
STEPS = 300
RESTARTS = 3  # fixed optimization budget; selected only by penalized training loss
PENALTY = RankOnePenalty(0.02, 0.2, 0.001)
DOMAIN_AREA = 9600.0  # km^2
SCALE = torch.tensor([60000., 40000.], dtype=torch.float64)


class SyntheticTruth:
    beta = torch.tensor([0.3, -0.2], dtype=torch.float64)
    u = torch.tensor([1.5, 0.8], dtype=torch.float64)

    @staticmethod
    def h(xy):
        return torch.tanh(1.5 * xy[:, 0] / 60000.)

    def local_coefficients(self, xy):
        return self.beta + self.h(xy)[:, None] * self.u

    def g(self, xy):
        return ((xy / SCALE) * self.local_coefficients(xy)).sum(1)

    def sample(self, n, seed):
        """Continuous rejection sampling: PPP conditional on prescribed N=n.

        Uniform proposal over domain, exact acceptance exp(g-M), with analytic
        bound M=sum|beta|+sum|u| since |x_j|, |h|<=1. No grid snapping or labels.
        Fixed batch size also makes the n=44 sample a prefix of n=256 per seed.
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = torch.Generator().manual_seed(seed)
        bound = self.beta.abs().sum() + self.u.abs().sum()
        accepted = []
        count = 0
        while count < n:
            draws = torch.rand((2048, 3), generator=rng, dtype=torch.float64)
            xy = (2 * draws[:, :2] - 1) * SCALE
            keep = draws[:, 2].log() <= self.g(xy) - bound
            accepted.append(xy[keep])
            count += int(keep.sum())
        return torch.cat(accepted)[:n]


def quadrature(resolution):
    """Midpoints of mildly nonuniform rectangular cells, true km^2 areas."""
    if not isinstance(resolution, int) or resolution < 2:
        raise ValueError("resolution must be an integer >=2")
    t = torch.linspace(-1, 1, resolution + 1, dtype=torch.float64)
    edges = t + 0.12 * torch.sin(math.pi * t)
    mid = (edges[:-1] + edges[1:]) / 2
    xx, yy = torch.meshgrid(mid, mid, indexing="ij")
    xy = torch.stack((xx.flatten(), yy.flatten()), 1) * SCALE
    width = edges[1:] - edges[:-1]
    area = (width[:, None] * width[None, :]).flatten() * DOMAIN_AREA / 4
    return xy, area


def normalized_density(g, area):
    log_z = torch.logsumexp(area.log() + g, 0)
    return (g - log_z).exp(), log_z


def correlation(a, b, weights):
    w = weights / weights.sum()
    a, b = a - (w * a).sum(), b - (w * b).sum()
    denom = ((w * a.square()).sum() * (w * b.square()).sum()).sqrt()
    return float((w * a * b).sum() / denom) if float(denom) > 1e-20 else 0.0


def rmse(a, b, area):
    return float(((a - b).square() * area / area.sum()).sum().sqrt())


@torch.no_grad()
def evaluate(model, truth, xy, area):
    fitted, target = model.g(xy), truth.g(xy)
    density, log_z = normalized_density(fitted, area)
    truth_density, truth_log_z = normalized_density(target, area)
    # Relative intensity A*p has area mean 1, avoiding tiny km^-2 RMSE values.
    relative, true_relative = density * DOMAIN_AREA, truth_density * DOMAIN_AREA
    fitted_beta, true_beta = model.local_coefficients(xy), truth.local_coefficients(xy)
    def top_mask(values):
        order = torch.argsort(values, descending=True)
        cumulative = area[order].cumsum(0)
        mask = torch.zeros(len(values), dtype=torch.bool)
        mask[order[cumulative - area[order] < .2 * area.sum()]] = True
        return mask
    a, b = top_mask(relative), top_mask(true_relative)
    deviations = fitted_beta - model.beta
    singular = torch.linalg.svdvals(deviations)
    metrics = dict(g_correlation=correlation(fitted, target, area),
                   g_rmse=rmse(fitted, target, area),
                   intensity_correlation=correlation(relative, true_relative, area),
                   intensity_rmse=rmse(relative, true_relative, area),
                   density_rmse=rmse(density, truth_density, area),
                   top20_overlap=float(area[a & b].sum() / area[b].sum()),
                   coefficient_rmse=float(((fitted_beta - true_beta).square()
                                            * (area / area.sum())[:, None]).mean(1).sum().sqrt()),
                   coefficient_correlations=[correlation(fitted_beta[:, j], true_beta[:, j], area)
                                             for j in range(2)],
                   reference_error=float(model.h(model.reference).abs().max()),
                   rank_one_residual=float(singular[1] / singular[0]) if singular[0] > 0 else 0.,
                   evaluation_log_z=float(log_z), truth_log_z=float(truth_log_z))
    return metrics, relative


def fit_case(n, seed, resolution=12):
    truth = SyntheticTruth()
    events = truth.sample(n, seed)
    q, area = quadrature(resolution)
    candidates = []
    for restart in range(RESTARTS):
        model = ReducedRankOneSyntheticPredictor(seed=seed + 1000 + 10000 * restart)
        history = fit_rank_one_synthetic(model, events, q, area, penalty=PENALTY,
                                        steps=STEPS, learning_rate=.01)
        candidates.append((model, history))
    selected = min(range(RESTARTS), key=lambda k: candidates[k][1][-1]['objective'])
    # Truth evaluation occurs AFTER model selection, never in the optimizer.
    grid, weights = quadrature(48)
    diagnostics = []
    for model, history in candidates:
        metrics, _ = evaluate(model, truth, grid, weights)
        metrics.update(n=n, seed=seed, resolution=resolution,
                       initial=history[0], final=history[-1])
        diagnostics.append(metrics)
    model = candidates[selected][0]
    metrics = dict(diagnostics[selected])
    metrics.update(selected_restart=selected, starts=diagnostics)
    _, surface = evaluate(model, truth, grid, weights)
    return metrics, surface, model



def stability_summary(cases, surfaces, area):
    summary = {}
    for key in ("intensity_correlation", "intensity_rmse", "g_correlation", "g_rmse",
                "coefficient_rmse", "top20_overlap"):
        values = torch.tensor([case[key] for case in cases], dtype=torch.float64)
        summary[key] = dict(mean=float(values.mean()), std=float(values.std(correction=1)))
    values = torch.tensor([case["final"]["objective"] for case in cases], dtype=torch.float64)
    summary["final_objective"] = dict(mean=float(values.mean()), std=float(values.std(correction=1)))
    pairs = list(itertools.combinations(surfaces, 2))
    summary["pairwise_intensity_correlation_mean"] = sum(correlation(a, b, area) for a, b in pairs) / len(pairs)
    summary["pairwise_intensity_rmse_mean"] = sum(rmse(a, b, area) for a, b in pairs) / len(pairs)
    return summary
