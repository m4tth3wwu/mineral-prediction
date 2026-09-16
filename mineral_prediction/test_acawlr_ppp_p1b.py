"""P1-B acceptance checks and reproducible, synthetic-only experiment report.

Broad recovery gates are fixed before the multi-seed run. Small-n degradation
is measured, not forced by an assertion. No parameter-level truth matching.
"""
import builtins
import copy
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from acawlr_ppp_bridge import (ReducedRankOneSyntheticPredictor, fit_rank_one_synthetic,
                               rank_one_value_and_grad)
from acawlr_ppp_synthetic import (SyntheticTruth, SEEDS, STEPS, RESTARTS, PENALTY, SCALE,
                                 DOMAIN_AREA, quadrature, normalized_density,
                                 fit_case, stability_summary, evaluate,
                                 correlation, rmse)


@contextmanager
def synthetic_io_only():
    """Deny Python filesystem reads/writes except the existing pure-core adapter.

    Applied throughout all recovery/refinement fits; report writing is outside.
    Also tests deny arbitrary data paths rather than relying on naming alone.
    """
    allowed = Path(__file__).resolve().parents[1] / 'scripts' / 'ACAWLR_improved.py'
    original_builtin, original_io = builtins.open, io.open
    reads = []
    def guarded(original):
        def open_file(file, mode='r', *args, **kwargs):
            if not isinstance(file, (str, bytes, os.PathLike)):
                raise AssertionError('Unexpected file descriptor I/O')
            path = Path(file).resolve()
            if path != allowed or any(flag in mode for flag in 'wax+'):
                raise AssertionError('Non-synthetic file access forbidden: ' + str(path))
            reads.append(str(path))
            return original(file, mode, *args, **kwargs)
        return open_file
    with patch('builtins.open', guarded(original_builtin)), patch('io.open', guarded(original_io)):
        yield reads


class SyntheticP1BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        # Load optimizer implementation before filesystem guard, no training/data.
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        cls.runs, cls.surfaces, cls.models = {}, {}, {}
        cls.truth = SyntheticTruth()
        cls.grid, cls.area = quadrature(48)
        with synthetic_io_only() as reads:
            for n in (256, 44):
                cls.runs[n], cls.surfaces[n], cls.models[n] = [], [], []
                for seed in SEEDS:
                    metrics, surface, model = fit_case(n, seed)
                    cls.runs[n].append(metrics)
                    cls.surfaces[n].append(surface)
                    cls.models[n].append(model)
                    print(f'P1-B n={n} seed={seed}: intensity corr={metrics["intensity_correlation"]:.4f}, '
                          f'RMSE={metrics["intensity_rmse"]:.4f}', flush=True)
            cls.refinement = [cls.runs[256][0]]
            cls.refinement_surfaces = [cls.surfaces[256][0]]
            for resolution in (18, 24):
                metrics, surface, _ = fit_case(256, SEEDS[0], resolution)
                cls.refinement.append(metrics)
                cls.refinement_surfaces.append(surface)
                print(f'P1-B refinement {resolution}x{resolution}: '
                      f'corr={metrics["intensity_correlation"]:.4f}', flush=True)
        cls.allowed_reads = reads
        cls.summaries = {n: stability_summary(cls.runs[n], cls.surfaces[n], cls.area)
                         for n in (256, 44)}
        cls.fixed_refinement = []
        with torch.no_grad():
            for resolution in (12, 24, 48):
                q, area = quadrature(resolution)
                model = cls.models[256][0]
                _, log_z = normalized_density(model.g(q), area)
                _, truth_log_z = normalized_density(cls.truth.g(q), area)
                # Same fitted g and eval grid; only integration resolution changes.
                surface = DOMAIN_AREA * (model.g(cls.grid) - log_z).exp()
                cls.fixed_refinement.append(dict(resolution=resolution, log_z=float(log_z),
                    truth_log_z=float(truth_log_z),
                    intensity_rmse=rmse(surface, cls.surfaces[256][0], cls.area)))

    @classmethod
    def tearDownClass(cls):
        report = dict(seeds=SEEDS, steps=STEPS, restarts=RESTARTS, learning_rate=.01,
                      penalty=vars(PENALTY), dtype='float64', area_unit='km^2',
                      intensity_rmse_unit='relative intensity A*p; area mean 1',
                      std='sample standard deviation, ddof=1',
                      runs=cls.runs, summaries=cls.summaries,
                      fixed_model_refinement=cls.fixed_refinement,
                      refit_refinement=cls.refinement,
                      refit_surface_comparisons=[dict(
                          resolution=case['resolution'],
                          correlation_to_12=correlation(surface, cls.refinement_surfaces[0], cls.area),
                          rmse_to_12=rmse(surface, cls.refinement_surfaces[0], cls.area))
                          for case, surface in zip(cls.refinement, cls.refinement_surfaces)])
        print('P1-B summaries: ' + json.dumps(cls.summaries), flush=True)
        output = os.environ.get('ACAWLR_P1B_REPORT')
        if output:
            Path(output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    def test_generator_reproducibility_and_domain(self):
        a = self.truth.sample(256, 11)
        torch.testing.assert_close(a, self.truth.sample(256, 11), rtol=0, atol=0)
        torch.testing.assert_close(a[:44], self.truth.sample(44, 11), rtol=0, atol=0)
        self.assertFalse(torch.equal(a, self.truth.sample(256, 23)))
        self.assertTrue(bool(torch.isfinite(a).all()))
        self.assertTrue(bool((a.abs() < SCALE).all()))
        # Continuous sampling: draws do not coincide with quadrature locations.
        self.assertFalse(bool((a[:, None, :] == self.grid[None, :, :]).all(2).any()))

    def test_truth_normalization_and_nontrivial_structure(self):
        density, log_z = normalized_density(self.truth.g(self.grid), self.area)
        torch.testing.assert_close((density * self.area).sum(), density.new_tensor(1.))
        self.assertTrue(bool(torch.isfinite(log_z)))
        linear = (self.grid / SCALE) @ self.truth.beta
        self.assertGreater(rmse(self.truth.g(self.grid), linear, self.area), .3)
        # Independent continuous sampler moment check on a larger synthetic draw.
        sample = self.truth.sample(20000, 91)
        expected = ((self.grid / SCALE) * (density * self.area)[:, None]).sum(0)
        torch.testing.assert_close((sample / SCALE).mean(0), expected, rtol=0, atol=.025)

    def test_n256_truth_recovery(self):
        # Broad aggregate thresholds; no best-seed reporting or perfect recovery.
        summary = self.summaries[256]
        self.assertGreater(summary['intensity_correlation']['mean'], .7)
        self.assertLess(summary['intensity_rmse']['mean'], .8)
        self.assertGreater(summary['g_correlation']['mean'], .65)
        self.assertGreater(summary['top20_overlap']['mean'], .5)
        for case in self.runs[256]:
            self.assertGreater(case['intensity_correlation'], .4)

    def test_n44_smoke_and_stability_summary(self):
        for n in (256, 44):
            self.assertEqual(len(self.runs[n]), 5)
            self.assertEqual([r['seed'] for r in self.runs[n]], list(SEEDS))
            for case in self.runs[n]:
                self.assertLess(case['final']['objective'], case['initial']['objective'])
                self.assertLess(case['final']['profile_objective'], case['initial']['profile_objective'])
                for key in ('intensity_correlation', 'intensity_rmse', 'g_rmse', 'coefficient_rmse'):
                    self.assertTrue(torch.isfinite(torch.tensor(case[key])))
                for key in ('objective', 'penalty', 'log_z', 'intercept'):
                    self.assertTrue(torch.isfinite(torch.tensor(case['final'][key])))
            values = torch.tensor([r['intensity_correlation'] for r in self.runs[n]], dtype=torch.float64)
            self.assertAlmostEqual(float(values.std(correction=1)),
                                   self.summaries[n]['intensity_correlation']['std'])
            self.assertTrue(-1 <= self.summaries[n]['pairwise_intensity_correlation_mean'] <= 1)

    def test_quadrature_refinement_sanity(self):
        fixed = self.fixed_refinement
        self.assertLess(abs(fixed[-1]['log_z'] - fixed[-2]['log_z']), .03)
        self.assertLess(abs(fixed[-1]['truth_log_z'] - fixed[-2]['truth_log_z']), .01)
        self.assertLess(fixed[1]['intensity_rmse'], fixed[0]['intensity_rmse'])
        base = self.refinement[0]
        for case, surface in zip(self.refinement[1:], self.refinement_surfaces[1:]):
            self.assertLess(abs(case['intensity_correlation'] - base['intensity_correlation']), .15)
            self.assertLess(abs(case['intensity_rmse'] - base['intensity_rmse']), .35)
            self.assertGreater(correlation(surface, self.refinement_surfaces[0], self.area), .85)

    def test_reference_rank_one_and_profile_mass(self):
        torch.testing.assert_close(self.truth.h(torch.zeros((1, 2), dtype=torch.float64)),
                                   torch.zeros(1, dtype=torch.float64), rtol=0, atol=0)
        for n in (256, 44):
            for case, model in zip(self.runs[n], self.models[n]):
                self.assertEqual(case['reference_error'], 0.)
                self.assertLess(case['rank_one_residual'], 1e-12)
                with torch.no_grad():
                    q, area = quadrature(case['resolution'])
                    mass = (area * (case['final']['intercept'] + model.g(q)).exp()).sum()
                torch.testing.assert_close(mass, mass.new_tensor(float(n)), rtol=1e-12, atol=1e-10)

    def test_full_streaming_gradients_and_updates(self):
        # Realized synthetic sample + already-trained nonlinear state, not toy rows.
        events = self.truth.sample(44, SEEDS[0])
        q, area = quadrature(12)
        model = copy.deepcopy(self.models[44][0])
        full = rank_one_value_and_grad(model, events, q, area, PENALTY)
        for chunk in (17, 59):
            stream = rank_one_value_and_grad(model, events, q, area, PENALTY, chunk)
            torch.testing.assert_close(full[2], stream[2], rtol=1e-12, atol=1e-10)
            for a, b in zip(full[3], stream[3]):
                self.assertTrue(bool(torch.isfinite(a).all() & torch.isfinite(b).all()))
                torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
        other = copy.deepcopy(model)
        kwargs = dict(penalty=PENALTY, steps=3, learning_rate=.001)
        fit_rank_one_synthetic(model, events, q, area, **kwargs)
        fit_rank_one_synthetic(other, events, q, area, chunk_size=17, **kwargs)
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)

    def test_restart_selection_uses_training_objective_only(self):
        for n in (256, 44):
            for case in self.runs[n]:
                self.assertEqual(len(case['starts']), RESTARTS)
                objectives = [run['final']['objective'] for run in case['starts']]
                self.assertEqual(case['final']['objective'], min(objectives))
                self.assertEqual(case['selected_restart'], objectives.index(min(objectives)))

    def test_no_real_data_io(self):
        self.assertGreater(len(self.allowed_reads), 0)
        self.assertEqual(len(set(self.allowed_reads)), 1)
        with synthetic_io_only():
            with self.assertRaises(AssertionError):
                Path('forbidden-real-data.csv').read_text()
            with self.assertRaises(AssertionError):
                builtins.open('forbidden-real-data.csv')
            sample = self.truth.sample(44, 1)
            q, area = quadrature(4)
            fit_rank_one_synthetic(ReducedRankOneSyntheticPredictor(), sample, q, area,
                                   penalty=PENALTY, steps=1)


if __name__ == '__main__':
    unittest.main()
