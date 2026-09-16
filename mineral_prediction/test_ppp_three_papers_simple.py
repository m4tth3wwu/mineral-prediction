"""Small deterministic checks for the exact-event PPP and validation rules."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

import PPP_three_papers_simple as model


class PointProcessTests(unittest.TestCase):
    def setUp(self):
        self.grid = pd.DataFrame({"x": [-1., 0., 1.], "area_km2": [1., 2., 3.]})
        self.events = pd.DataFrame({"x": [0.2, 0.6]})

    def test_gradient(self):
        q = np.array([[-1., 0.], [0.3, 1.], [1., 2.]])
        e = np.array([0.5, 1.5])
        args = (q, e, np.log([1., 2., 3.]), 2, 0.1)
        error = check_grad(lambda b: model.profiled_objective(b, *args)[0],
                           lambda b: model.profiled_objective(b, *args)[1], np.array([0.3, -0.2]))
        self.assertLess(error, 1e-6)

    def test_mass_equals_event_count(self):
        fit = model.fit_points(self.grid, self.events, ["x"])
        mass = np.sum(self.grid.area_km2 * np.exp(model.log_intensity(fit, self.grid)))
        self.assertAlmostEqual(mass, 2., places=9)
        ll = model.log_intensity(fit, self.events).sum() - mass
        self.assertAlmostEqual(ll, fit.log_likelihood, places=9)

    def test_event_features_not_grid_counts(self):
        self.grid["event_count"] = [100, 500, 1000]
        first = model.fit_points(self.grid, self.events, ["x"])
        self.grid["event_count"] = [0, 0, 0]
        unchanged = model.fit_points(self.grid, self.events, ["x"])
        np.testing.assert_allclose(first.slopes, unchanged.slopes)
        shifted = model.fit_points(self.grid, pd.DataFrame({"x": [-0.9, -0.8]}), ["x"])
        self.assertGreater(abs(first.slopes[0] - shifted.slopes[0]), 0.5)

    def test_area_units_only_change_intercept(self):
        fit = model.fit_points(self.grid, self.events, ["x"])
        larger = self.grid.copy()
        larger.area_km2 *= 100
        other = model.fit_points(larger, self.events, ["x"])
        np.testing.assert_allclose(fit.slopes, other.slopes, atol=1e-6)
        self.assertAlmostEqual(other.intercept, fit.intercept - np.log(100), places=6)

    def test_constant_intensity_scores(self):
        self.grid["x"] = 1.
        self.events["x"] = 1.
        fit = model.fit_points(self.grid, self.events, ["x"])
        scores = model.evaluate(fit, self.grid, self.events)
        self.assertAlmostEqual(scores["presence_background_auc"], 0.5)
        self.assertAlmostEqual(scores["conditional_log_gain_per_event"], 0.)
        # Ties cannot legitimately be described as covering exactly 10%.
        self.assertEqual(scores["top_10pct_actual_area_fraction"], 1.)

    def test_preprocessing_uses_training_domain_only(self):
        prep = model.preprocess(self.grid, ["x"])
        self.assertAlmostEqual(prep.means["x"], 1 / 3)
        before = dict(prep.means)
        model.matrix(pd.DataFrame({"x": [1e20, np.nan]}), prep)
        self.assertEqual(prep.means, before)

    def test_coverage_standardization_is_explicit(self):
        grid = self.grid.rename(columns={"x": model.PROXY})
        events = self.events.rename(columns={"x": model.PROXY})
        fit = model.fit_points(grid, events, [model.PROXY])
        controlled = model.log_intensity(fit, grid, True)
        np.testing.assert_allclose(controlled, fit.intercept)
        self.assertGreater(np.ptp(model.log_intensity(fit, grid)), 0)

    def test_logsumexp_does_not_overflow(self):
        val, grad = model.profiled_objective(np.array([1000.]), np.array([[-1.], [0.], [1.]]),
                                              np.array([0.]), np.zeros(3), 2, 0.1)
        self.assertTrue(np.isfinite(val) and np.isfinite(grad).all())


class SpatialTests(unittest.TestCase):
    def test_fixed_domain_coarse_areas(self):
        fine = pd.DataFrame({"metric_x": [2500, 7500, 12500], "metric_y": [2500] * 3,
                             "area_km2": [25.] * 3, "cell_id": [0, 1, 2], "event_count": [1, 0, 0]})
        coarse = model.coarsen_quadrature(fine, 10)
        self.assertEqual(coarse.area_km2.tolist(), [50., 25.])
        self.assertEqual(coarse.area_km2.sum(), fine.area_km2.sum())
        self.assertNotIn("event_count", coarse)
        self.assertTrue(set(coarse.metric_x).issubset(set(fine.metric_x)))

    def test_guard_uses_distance_to_square_not_centre(self):
        frame = pd.DataFrame({"metric_x": [201000, 251000, 230000, 240000],
                              "metric_y": [100000, 100000, 230000, 240000]})
        # Distance to a 200 km test block: 1, 51, sqrt(30²+30²), sqrt(40²+40²).
        np.testing.assert_array_equal(model.outside_guard(frame, [(0, 0)], 200, 50),
                                      [False, True, False, True])

    def test_fold_membership_depends_on_real_coordinates(self):
        grid = pd.DataFrame({"metric_x": [2500, 7500, 12500, 17500, 22500, 27500],
                             "metric_y": [2500] * 6, "area_km2": [25.] * 6, "cell_id": range(6)})
        events = pd.DataFrame({"metric_x": [9999, 10001, 22000], "metric_y": [2000] * 3})
        mapping = model.make_fold_map(grid, events, 10, 3, 42)
        ids = model.fold_ids(events, mapping, 10)
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(mapping, model.make_fold_map(grid, events, 10, 3, 42))


@unittest.skipUnless((model.ROOT / "output/ppp_three_papers_simple_v2/README_results.md").exists(),
                     "Run the real-data experiment first")
class RealDataResultTests(unittest.TestCase):
    directory = model.ROOT / "output/ppp_three_papers_simple_v2"

    def test_every_training_event_held_out_once_per_repeat(self):
        held = pd.read_csv(self.directory / "held_out_event_predictions.csv")
        groups = held.groupby(["model", "seed"])
        self.assertEqual(len(groups), 12)
        self.assertTrue((groups.size() == 44).all())
        self.assertTrue((groups.deposit.nunique() == 44).all())
        self.assertTrue(np.isfinite(held.log_intensity).all())

    def test_buffer_and_optimizer_audit(self):
        for file, expected_rows in (("spatial_fold_metrics.csv", 60), ("fine_spatial_fold_metrics.csv", 20)):
            rows = pd.read_csv(self.directory / file)
            self.assertEqual(len(rows), expected_rows)
            self.assertTrue((rows.minimum_event_separation_km > 50).all())
            self.assertTrue((rows.optimizer_gradient_max < 0.01).all())
            self.assertTrue((rows.groupby(["model", "seed"]).test_events.sum() == 44).all())

    def test_external_and_prediction_outputs_are_complete(self):
        external = pd.read_csv(self.directory / "external_deposit_scores.csv")
        self.assertEqual(len(external), 64)
        self.assertTrue((external.groupby("model").DEPOSIT.nunique() == 16).all())
        predictions = pd.read_csv(self.directory / "predictions_5km.csv")
        self.assertEqual(len(predictions), 132888)
        cols = [c for c in predictions if c.endswith("intensity")]
        self.assertEqual(len(cols), 5)
        self.assertTrue(np.isfinite(predictions[cols]).all().all())
        self.assertTrue((predictions[cols] > 0).all().all())

    def test_integration_precision_on_this_dataset(self):
        convergence = pd.read_csv(self.directory / "quadrature_convergence.csv")
        self.assertEqual(len(convergence), 4)
        self.assertTrue((convergence.coarse_integral_relative_error.abs() < 0.05).all())
        self.assertTrue((convergence.map_spearman > 0.99).all())


if __name__ == "__main__":
    unittest.main()
