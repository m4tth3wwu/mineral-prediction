"""Synthetic-only ACAWLR-style PPP bridge; no fitting or real-data I/O.

Reuses the core's pure neural classes without executing its GIS/config/entrypoint.
All neural weights are prescribed constants, BatchNorm stays in eval mode.
Coordinates/anchors are synthetic metres; quadrature areas are km^2.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import math

import torch
from torch import nn


def _pure_core_classes():
    """Minimal source adapter: load only these four class definitions."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "ACAWLR_improved.py"
    names = {"ASPNN", "ResBlock", "CBAM", "CAWNN"}
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in classes} != names:
        raise RuntimeError("The selected ACAWLR pure-class interface has changed")
    namespace = {"torch": torch, "nn": nn, "__name__": __name__}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace["ASPNN"], namespace["CAWNN"]


class SyntheticSpatialPredictor(nn.Module):
    """P0 fixed spatial function, not a geological research model.

    g(s, gain) = gain * (t_fixed(s) - t_fixed(reference)).
    The scalar gain and input coordinates may carry gradients for diagnostics;
    no optimizer or parameter update is provided. Constant synthetic covariates
    deliberately isolate location effects in the neural weighting path.
    """

    def __init__(self):
        super().__init__()
        ASPNN, CAWNN = _pure_core_classes()
        with torch.random.fork_rng(devices=[]):
            self.aspnn = ASPNN().double()
            self.cawnn = CAWNN(coef_num=2).double()
        # Hand-prescribed fixture, independent of events, labels and outcomes.
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    fan_in = module.weight[0].numel()
                    module.weight.fill_(0.8 / fan_in)
                    if module.bias is not None:
                        module.bias.fill_(0.2)
        axes = torch.meshgrid(torch.linspace(-60000, 60000, 4, dtype=torch.float64),
                              torch.linspace(-40000, 40000, 3, dtype=torch.float64),
                              indexing="ij")
        self.register_buffer("anchors", torch.stack(axes, dim=-1))
        self.register_buffer("reference", torch.zeros((1, 2), dtype=torch.float64))
        self.register_buffer("features", torch.tensor([1.0, -0.25],
                                                      dtype=torch.float64))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # P0 never estimates BN statistics, even if an enclosing caller uses train().
        return super().train(False)

    def _raw(self, xy):
        delta = self.anchors[None] - xy[:, None, None, :]
        angle = 0.35  # fixed synthetic direction, radians
        parallel = (delta[..., 0] * math.cos(angle)
                    + delta[..., 1] * math.sin(angle)) / 150000.0
        perpendicular = (-delta[..., 0] * math.sin(angle)
                         + delta[..., 1] * math.cos(angle)) / 50000.0
        offsets = torch.stack((parallel, perpendicular), dim=-1).clamp(-20, 20)
        proximity = self.aspnn(offsets.reshape(-1, 2))
        gaspg = proximity.reshape(len(xy), 1, 4, 3).clamp(1e-6, 1 - 1e-6)
        local_weights = self.cawnn(gaspg).clamp(-50, 50)
        # No binary global_beta, final predictor clamp, or final sigmoid.
        return (local_weights * self.features).sum(dim=1)

    def g(self, xy, gain=1.0):
        """The only score function, shared by events and quadrature."""
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
            raise ValueError("Need nonempty synthetic coordinates of shape (N, 2)")
        if not torch.isfinite(xy).all():
            raise ValueError("Coordinates must be finite")
        return gain * (self._raw(xy) - self._raw(self.reference))


@dataclass(frozen=True)
class ProfiledPPP:
    objective: torch.Tensor
    log_z: torch.Tensor
    intercept: torch.Tensor


def _validate(events, quadrature, area):
    if len(events) == 0 or len(quadrature) == 0:
        raise ValueError("Need events and a nonempty integration domain")
    if area.ndim != 1 or len(area) != len(quadrature):
        raise ValueError("Need one area per quadrature point")
    if not torch.isfinite(area).all() or not (area > 0).all():
        raise ValueError("Areas must be finite and strictly positive")


def profiled_ppp(model, events, quadrature, area, gain):
    """n*logZ - sum_event g; parameter-independent constants omitted, no penalty."""
    _validate(events, quadrature, area)
    event_scores = model.g(events, gain)
    q_scores = model.g(quadrature, gain)
    log_z = torch.logsumexp(area.log() + q_scores, dim=0)
    n = len(events)
    return ProfiledPPP(n * log_z - event_scores.sum(),
                       log_z, math.log(n) - log_z)


def streaming_ppp_value_and_grad(model, events, quadrature, area, gain,
                                 chunk_size, wrt=None):
    """Exact two-pass discrete integral and gradient, NOT minibatch profiling.

    First pass accumulates logZ with no graph. Second pass differentiates
    n*sum stop_gradient(p_q)*g(q) and -sum g(event), one chunk at a time.
    Returned objective is the true objective, not the gradient surrogate.
    Inputs/state must stay fixed throughout both passes. Returned gradients are
    diagnostics only (no .grad accumulation/update; no higher-order graph).
    """
    _validate(events, quadrature, area)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    wrt = (gain,) if wrt is None else tuple(wrt)
    gradients = [torch.zeros_like(item) for item in wrt]
    n = len(events)

    def accumulate(term):
        pieces = torch.autograd.grad(term, wrt, allow_unused=True)
        for index, piece in enumerate(pieces):
            if piece is not None:
                gradients[index] += piece.detach()

    with torch.no_grad():
        log_z = area.new_tensor(-float("inf"))
        for start in range(0, len(quadrature), chunk_size):
            stop = start + chunk_size
            scores = model.g(quadrature[start:stop], gain)
            part = torch.logsumexp(area[start:stop].log() + scores, dim=0)
            log_z = torch.logaddexp(log_z, part)

    for start in range(0, len(quadrature), chunk_size):
        stop = start + chunk_size
        scores = model.g(quadrature[start:stop], gain)
        probabilities = (area[start:stop].log() + scores.detach() - log_z).exp()
        accumulate(n * (probabilities * scores).sum())

    event_sum = area.new_zeros(())
    for start in range(0, n, chunk_size):
        scores = model.g(events[start:start + chunk_size], gain)
        event_sum += scores.detach().sum()
        accumulate(-scores.sum())

    result = ProfiledPPP(n * log_z - event_sum, log_z, math.log(n) - log_z)
    return result, tuple(gradients)
