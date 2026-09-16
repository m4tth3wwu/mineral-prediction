"""P1-A synthetic-only tests: python -B -m unittest -v test_acawlr_ppp_p1."""
import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from acawlr_ppp_bridge import (
    ReducedRankOneSyntheticPredictor, RankOnePenalty, profiled_ppp,
    rank_one_value_and_grad, fit_rank_one_synthetic,
)


class SyntheticP1ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.model = ReducedRankOneSyntheticPredictor()
        self.events = torch.tensor([[-32000, -17000], [13000, 21000],
                                    [47000, -9000]], dtype=torch.float64)
        self.q = torch.tensor([[-50000, -30000], [-27000, 12000], [0, -22000],
                               [17000, 31000], [38000, -8000], [55000, 25000]],
                              dtype=torch.float64)
        self.area = torch.tensor([0.5, 2.0, 1.25, 3.0, 0.75, 4.0], dtype=torch.float64)
        # Test choices only, not scientifically selected hyperparameters.
        self.penalty = RankOnePenalty(beta=0.02, u=0.2, theta=0.001)

    def test_architecture_rank_one_and_reference_constraint(self):
        m = self.model
        self.assertEqual(sum(p.numel() for p in m.parameters()), 675)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        self.assertFalse(any(isinstance(x, (nn.modules.batchnorm._BatchNorm, nn.Dropout))
                             for x in m.modules()))
        self.assertIsNone(m.cawnn.readout.bias)
        h = m.h(self.q)
        self.assertGreater(float(h.detach().max() - h.detach().min()), 1e-7)
        self.assertTrue(bool((h.abs() <= 2).all()))
        torch.testing.assert_close(m.h(m.reference), torch.zeros(1, dtype=torch.float64),
                                   rtol=0, atol=0)
        deviations = m.local_coefficients(self.q) - m.beta
        torch.testing.assert_close(deviations, h[:, None] * m.u)
        self.assertEqual(int(torch.linalg.matrix_rank(deviations)), 1)
        # Constant covariates isolate the spatial route, excluding x(s) variation.
        with patch.object(m, "features", side_effect=lambda xy: xy.new_ones((len(xy), 2))):
            scores = m.g(self.q)
            self.assertGreater(float((scores.max() - scores.min()).detach()), 1e-8)
        with torch.no_grad():
            m.u.zero_()
            m.beta.copy_(torch.tensor([0.3, -0.1]))
        torch.testing.assert_close(m.g(self.q), m.features(self.q) @ m.beta)
        # No final sigmoid/clamp: legitimate raw scores may exceed binary ranges.
        with torch.no_grad():
            m.beta.fill_(100.)
        self.assertGreater(float(m.g(self.q).detach().max()), 30.)

    def test_role_chunk_reorder_and_train_eval_invariance(self):
        m = self.model
        xy = torch.cat((self.events, self.q))
        before = copy.deepcopy(m.state_dict())
        for training in (True, False):
            m.train(training)
            expected = m.g(xy)
            order = torch.tensor([8, 2, 0, 6, 4, 1, 7, 3, 5])
            for chunk in (1, 2, 4, 20):
                actual = torch.cat([m.g(part) for part in xy[order].split(chunk)])
                torch.testing.assert_close(actual, expected[order], rtol=1e-10, atol=1e-13)
        for key, value in m.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        observed = []
        original = m.g
        def capture(xy, gain):
            out = original(xy, gain)
            observed.append(out)
            return out
        with patch.object(m, "g", side_effect=capture):
            profiled_ppp(m, self.events, torch.cat((self.q, self.events[:1])),
                         torch.cat((self.area, self.area.new_tensor([1.1]))), 1.)
        self.assertEqual(len(observed), 2)
        torch.testing.assert_close(observed[0][0], observed[1][-1], rtol=1e-10, atol=1e-13)

    def test_axial_symmetry(self):
        m = self.model
        torch.testing.assert_close(m.raw_spatial(self.q, m.angle),
                                   m.raw_spatial(self.q, m.angle + math.pi),
                                   rtol=1e-12, atol=1e-14)

    def test_all_parameter_gradient_blocks_against_finite_difference(self):
        # Likelihood alone: regularization cannot conceal a detached/dead trunk.
        m = self.model
        parameters = dict(m.named_parameters())
        loss = profiled_ppp(m, self.events, self.q, self.area, 1.).objective
        grads = dict(zip(parameters, torch.autograd.grad(loss, tuple(parameters.values()))))
        # Probe the largest derivative in every tensor (including biases/attention).
        for name, parameter in parameters.items():
            with self.subTest(parameter=name):
                grad = grads[name].reshape(-1)
                self.assertTrue(bool(torch.isfinite(grad).all()))
                index = int(grad.abs().argmax())
                self.assertGreater(float(grad[index].abs()), 1e-12)
                original = parameter.detach().clone()
                eps = 1e-5
                values = []
                try:
                    for sign in (1, -1):
                        with torch.no_grad():
                            parameter.copy_(original)
                            parameter.reshape(-1)[index] += sign * eps
                            values.append(float(profiled_ppp(
                                m, self.events, self.q, self.area, 1.).objective))
                finally:
                    with torch.no_grad():
                        parameter.copy_(original)
                numeric = (values[0] - values[1]) / (2 * eps)
                torch.testing.assert_close(grad[index], grad.new_tensor(numeric),
                                           rtol=5e-3, atol=2e-9)
        # The reference branch itself has nonzero parameter derivatives.
        ref_grads = torch.autograd.grad(m.raw_spatial(m.reference).tanh().sum(),
                                       tuple(m.aspnn.parameters()))
        self.assertGreater(sum(float(g.abs().sum()) for g in ref_grads), 1e-6)

    def test_full_streaming_penalty_and_gradient_agree(self):
        full, reg, total, gradients = rank_one_value_and_grad(
            self.model, self.events, self.q, self.area, self.penalty)
        for chunk in (1, 2, 4, 20):
            stream, sreg, stotal, sgrad = rank_one_value_and_grad(
                self.model, self.events.flip(0), self.q.flip(0), self.area.flip(0),
                self.penalty, chunk)
            for a, b in ((full.objective, stream.objective), (full.intercept, stream.intercept),
                         (full.log_z, stream.log_z), (reg, sreg), (total, stotal)):
                torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
            for a, b in zip(gradients, sgrad):
                torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-12)

    def test_training_decreases_loss_and_updates_neural_weights(self):
        initial = copy.deepcopy(self.model.state_dict())
        stream_model = copy.deepcopy(self.model)
        kwargs = dict(penalty=self.penalty, steps=40, learning_rate=0.01)
        full = fit_rank_one_synthetic(self.model, self.events, self.q, self.area, **kwargs)
        stream = fit_rank_one_synthetic(stream_model, self.events, self.q, self.area,
                                        chunk_size=2, **kwargs)
        self.assertLess(full[-1]["objective"], full[0]["objective"] - 0.01)
        self.assertLess(full[-1]["profile_objective"], full[0]["profile_objective"])
        for prefix in ("aspnn.", "cawnn.", "beta", "u"):
            self.assertTrue(any(not torch.equal(value, initial[key])
                                for key, value in self.model.state_dict().items()
                                if key.startswith(prefix)))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, stream_model.state_dict()[key], rtol=1e-7, atol=1e-9)
        for a, b in zip(full, stream):
            self.assertAlmostEqual(a["objective"], b["objective"], places=9)
            self.assertTrue(all(math.isfinite(v) for v in a.values() if isinstance(v, (int, float))))
        final = profiled_ppp(self.model, self.events, self.q, self.area, 1.)
        self.assertAlmostEqual(full[-1]["intercept"], float(final.intercept.detach()), places=12)
        mass = (self.area * (final.intercept + self.model.g(self.q)).exp()).sum()
        torch.testing.assert_close(mass, mass.new_tensor(float(len(self.events))))
        # Independent direct one-dimensional solution after actual training.
        low, high = -30., 30.
        gq = self.model.g(self.q).detach()
        for _ in range(100):
            middle = (low + high) / 2
            if float((self.area * (middle + gq).exp()).sum()) > len(self.events):
                high = middle
            else:
                low = middle
        self.assertAlmostEqual((low + high) / 2, full[-1]["intercept"], places=11)

    def test_invalid_inputs_and_nonfinite_state_fail(self):
        for events, area in ((self.events[:0], self.area),
                             (self.events, self.area * 0),
                             (self.events, self.area * float("nan"))):
            with self.assertRaises(ValueError):
                rank_one_value_and_grad(self.model, events, self.q, area, self.penalty)
        with self.assertRaises(ValueError):
            RankOnePenalty(0., -1., 0.)
        with torch.no_grad():
            self.model.beta.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            rank_one_value_and_grad(self.model, self.events, self.q, self.area, self.penalty)


if __name__ == "__main__":
    unittest.main()
