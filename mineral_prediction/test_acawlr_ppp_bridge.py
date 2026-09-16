"""Only synthetic P0 checks; run: python -m unittest -v test_acawlr_ppp_bridge."""
import math
import unittest
from unittest.mock import patch

import torch

from acawlr_ppp_bridge import (
    SyntheticSpatialPredictor, profiled_ppp, streaming_ppp_value_and_grad,
)


class SyntheticP0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)  # tiny CPU fixtures; restored after the suite

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.model = SyntheticSpatialPredictor()
        self.events = torch.tensor([[-32000, -17000], [13000, 21000],
                                    [47000, -9000]], dtype=torch.float64)
        self.q = torch.tensor([[-50000, -30000], [-27000, 12000], [0, -22000],
                               [17000, 31000], [38000, -8000], [55000, 25000]],
                              dtype=torch.float64)
        self.area = torch.tensor([0.5, 2.0, 1.25, 3.0, 0.75, 4.0],
                                 dtype=torch.float64)
        self.gain = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

    def test_event_quadrature_role_invariance(self):
        # Include an identical point in both roles; actual likelihood uses g twice.
        q = torch.cat((self.q, self.events[:1]))
        area = torch.cat((self.area, self.area.new_tensor([1.1])))
        observed = []
        original = self.model.g

        def capture(xy, gain):
            scores = original(xy, gain)
            observed.append(scores.detach())
            return scores

        with patch.object(self.model, "g", side_effect=capture) as called:
            profiled_ppp(self.model, self.events, q, area, self.gain)
        self.assertEqual(called.call_count, 2)
        torch.testing.assert_close(observed[0][0], observed[1][-1],
                                   rtol=1e-12, atol=1e-12)

    def test_chunk_size_and_reorder_invariance(self):
        state = {name: value.clone() for name, value in self.model.state_dict().items()}
        self.model.train()  # must not reactivate batch-dependent normalization
        self.assertFalse(any(m.training for m in self.model.modules()))
        self.assertFalse(any(p.requires_grad for p in self.model.parameters()))
        xy = torch.cat((self.events, self.q))
        expected = self.model.g(xy, self.gain)
        order = torch.tensor([8, 2, 0, 6, 4, 1, 7, 3, 5])
        for size in (1, 2, 4, 20):
            with self.subTest(chunk=size):
                actual = torch.cat([self.model.g(x, self.gain)
                                    for x in xy[order].split(size)])
                torch.testing.assert_close(actual, expected[order],
                                           rtol=1e-11, atol=1e-12)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_predictor_varies_with_position(self):
        # Features are identical everywhere: variation must come from spatial path.
        scores = self.model.g(self.q, self.gain)
        self.assertGreater(float((scores.max() - scores.min()).detach()), 1e-5)
        self.assertEqual(tuple(self.model.anchors.shape), (4, 3, 2))
        torch.testing.assert_close(self.model.g(self.model.reference, self.gain),
                                   torch.zeros(1, dtype=torch.float64))

    def test_profiled_intercept_matches_direct_one_dimensional_solution(self):
        result = profiled_ppp(self.model, self.events, self.q, self.area, self.gain)
        gq = self.model.g(self.q, self.gain).detach()
        ge = self.model.g(self.events, self.gain).detach()
        # Independent bisection of d(-log L)/db = integral exp(b+g) - n.
        low, high = -20.0, 20.0
        for _ in range(100):
            middle = (low + high) / 2
            score = float((self.area * (middle + gq).exp()).sum()) - len(ge)
            if score > 0:
                high = middle
            else:
                low = middle
        direct_b = (low + high) / 2
        self.assertAlmostEqual(float(result.intercept.detach()), direct_b, places=11)
        raw_nll = -len(ge) * direct_b - ge.sum() + (self.area * (direct_b + gq).exp()).sum()
        constant = len(ge) - len(ge) * math.log(len(ge))
        torch.testing.assert_close(raw_nll, result.objective.detach() + constant)
        torch.testing.assert_close((self.area * (result.intercept + gq).exp()).sum(),
                                   gq.new_tensor(float(len(ge))))

    def test_nonlinear_gradient_matches_finite_difference(self):
        events = self.events.clone().requires_grad_()
        q = self.q.clone().requires_grad_()
        objective = profiled_ppp(self.model, events, q, self.area, self.gain).objective
        gain_grad, event_grad, q_grad = torch.autograd.grad(
            objective, (self.gain, events, q))

        def loss(e, grid, gain):
            with torch.no_grad():
                return float(profiled_ppp(self.model, e, grid, self.area, gain).objective)

        h = 1e-3
        numeric = (loss(events, q, self.gain + h) -
                   loss(events, q, self.gain - h)) / (2 * h)
        self.assertGreater(abs(float(gain_grad)), 1e-7)
        torch.testing.assert_close(gain_grad, gain_grad.new_tensor(numeric),
                                   rtol=1e-5, atol=1e-9)
        # Coordinate probes differentiate THROUGH fixed ASPNN/CAWNN, not just gain.
        for role, index, gradient in (("event", (0, 0), event_grad),
                                      ("event", (1, 1), event_grad),
                                      ("q", (2, 0), q_grad),
                                      ("q", (4, 1), q_grad)):
            with self.subTest(role=role, index=index):
                ep, em = events.detach().clone(), events.detach().clone()
                qp, qm = q.detach().clone(), q.detach().clone()
                plus, minus = (ep, em) if role == "event" else (qp, qm)
                plus[index] += 0.1  # metres, safely away from fixture ReLU kinks
                minus[index] -= 0.1
                numeric = (loss(ep, qp, self.gain) - loss(em, qm, self.gain)) / 0.2
                self.assertGreater(abs(float(gradient[index])), 1e-10)
                torch.testing.assert_close(gradient[index],
                                           gradient.new_tensor(numeric),
                                           rtol=2e-4, atol=1e-11)

    def test_full_and_streaming_calculation_agree(self):
        events = self.events.clone().requires_grad_()
        q = self.q.clone().requires_grad_()
        wrt = (self.gain, events, q)
        full = profiled_ppp(self.model, events, q, self.area, self.gain)
        expected_grad = torch.autograd.grad(full.objective, wrt)
        for chunk in (1, 2, 4, 20):
            with self.subTest(chunk=chunk):
                stream, actual_grad = streaming_ppp_value_and_grad(
                    self.model, events, q, self.area, self.gain, chunk, wrt=wrt)
                for field in ("objective", "log_z", "intercept"):
                    torch.testing.assert_close(getattr(stream, field),
                                               getattr(full, field),
                                               rtol=1e-12, atol=1e-12)
                for actual, expected in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)

    def test_loss_and_gradient_are_finite(self):
        for value in (3.0, 1e8, -1e8):
            with self.subTest(gain=value):
                gain = torch.tensor(value, dtype=torch.float64, requires_grad=True)
                result = profiled_ppp(self.model, self.events, self.q, self.area, gain)
                gradient, = torch.autograd.grad(result.objective, (gain,))
                stream, stream_grad = streaming_ppp_value_and_grad(
                    self.model, self.events, self.q, self.area, gain, 2)
                for tensor in (result.objective, result.log_z, result.intercept,
                               gradient, stream.objective, stream_grad[0]):
                    self.assertTrue(bool(torch.isfinite(tensor).all()))
                torch.testing.assert_close(stream.objective, result.objective)
                torch.testing.assert_close(stream_grad[0], gradient, rtol=1e-8, atol=1e-10)
                if abs(value) > 1:
                    if abs(value) == 1e8:
                        self.assertGreater(float(self.model.g(self.q, gain).detach().abs().max()),
                                           800.0)  # exceeds safe naive float64 exp range


if __name__ == "__main__":
    unittest.main()
