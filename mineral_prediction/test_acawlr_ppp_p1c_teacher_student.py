"""P1-C checks; recovery failures remain ordinary failing unittest assertions."""
import copy
import inspect
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from acawlr_ppp_bridge import (ReducedRankOneSyntheticPredictor, profiled_ppp,
                               rank_one_value_and_grad, fit_rank_one_synthetic)
from acawlr_ppp_synthetic import quadrature, normalized_density, SCALE, PENALTY
from acawlr_ppp_teacher_student import (make_teacher, sample_events, prepare_design,
    exact_copy_sanity, teacher_diagnostics, fit_student, state_hash, run_experiments,
    save_report, refinement, rejection_bound)
from test_acawlr_ppp_p1b import synthetic_io_only


class TeacherStudentMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        cls.teacher = make_teacher()
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    def test_exact_representability_dense_grid(self):
        for value in exact_copy_sanity(self.teacher).values():
            self.assertLessEqual(value, 1e-12)
        student = ReducedRankOneSyntheticPredictor(91)
        student.load_state_dict(self.teacher.state_dict())
        self.assertEqual(type(student), type(self.teacher))
        self.assertEqual(state_hash(student.state_dict()), state_hash(self.teacher.state_dict()))

    def test_teacher_nontrivial_constraints_and_refinement(self):
        info = teacher_diagnostics(self.teacher)
        self.assertTrue(info['all_finite'])
        self.assertGreater(info['h']['std'], .02)
        self.assertGreater(info['spatial_score']['std'], .1)
        self.assertGreater(info['g']['std'], .15)
        self.assertLess(info['intensity_concentration']['max_relative'], 10)
        self.assertEqual(info['reference_error'], 0.)
        self.assertLess(info['rank_one_residual'], 1e-12)
        for comparison in info['refinement']['comparisons']:
            self.assertTrue(comparison['stable'])
        grid, area = quadrature(96)
        density, _ = normalized_density(self.teacher.g(grid), area)
        torch.testing.assert_close((density * area).sum(), area.new_tensor(1.))
        self.assertLess(float(self.teacher.g(grid).max()), rejection_bound(self.teacher))

    def test_continuous_sampling_reproducibility_prefix_and_moments(self):
        events = sample_events(self.teacher, 256, 11)
        torch.testing.assert_close(events, sample_events(self.teacher, 256, 11), rtol=0, atol=0)
        torch.testing.assert_close(events[:44], sample_events(self.teacher, 44, 11), rtol=0, atol=0)
        self.assertFalse(torch.equal(events, sample_events(self.teacher, 256, 23)))
        self.assertTrue(bool((events.abs() < SCALE).all()))
        self.assertTrue(bool(torch.isfinite(events).all()))
        for r in (24, 48, 96):
            grid, _ = quadrature(r)
            self.assertFalse(bool((events[:, None, :] == grid[None, :, :]).all(2).any()))
        # Independent sampler moment check: linear and quadratic coordinate moments.
        sample = sample_events(self.teacher, 10000, 91) / SCALE
        grid, area = quadrature(96)
        p, _ = normalized_density(self.teacher.g(grid), area)
        for power in (1, 2):
            expected = ((grid / SCALE).pow(power) * (p * area)[:, None]).sum(0)
            torch.testing.assert_close(sample.pow(power).mean(0), expected, rtol=0, atol=.035)

    def test_design_changes_exactly_one_source(self):
        design = prepare_design(self.teacher)
        for n in (44, 256):
            a = [c for c in design if c['experiment']=='A' and c['n']==n]
            b = [c for c in design if c['experiment']=='B' and c['n']==n]
            self.assertEqual(len(a), 5)
            self.assertEqual(len(b), 5)
            self.assertEqual(len({state_hash(c['initial_state']) for c in a}), 5)
            for c in a:
                torch.testing.assert_close(c['events'], a[0]['events'], rtol=0, atol=0)
            self.assertEqual(len({state_hash({'events':c['events']}) for c in b}), 5)
            for c in b:
                for key, value in c['initial_state'].items():
                    self.assertTrue(torch.equal(value, b[0]['initial_state'][key]))
            # Mutating global RNG does not alter the saved starting state.
            torch.rand(79)
            student = ReducedRankOneSyntheticPredictor(345)
            student.load_state_dict(b[0]['initial_state'])
            self.assertEqual(state_hash(student.state_dict()), state_hash(b[0]['initial_state']))
        for c in design:
            if c['n']==44:
                large = next(d for d in design if d['experiment']==c['experiment'] and
                             d['data_seed']==c['data_seed'] and d['init_seed']==c['init_seed'] and d['n']==256)
                self.assertTrue(torch.equal(c['events'], large['events'][:44]))

    def test_profile_objective_gradient_rank_reference_after_smoke(self):
        events = sample_events(self.teacher, 44, 11)
        initial = copy.deepcopy(ReducedRankOneSyntheticPredictor(1011).state_dict())
        model, _, grads = fit_student(events, initial, steps=2)
        self.assertTrue(grads['finite'])
        q, area = quadrature(24)
        full = rank_one_value_and_grad(model, events, q, area, PENALTY)
        stream = rank_one_value_and_grad(model, events, q, area, PENALTY, 67)
        torch.testing.assert_close(full[2], stream[2], rtol=1e-12, atol=1e-10)
        for a,b in zip(full[3],stream[3]):
            torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
        result = profiled_ppp(model, events, q, area, 1.)
        gq, ge = model.g(q), model.g(events)
        mass = (area * (result.intercept + gq).exp()).sum()
        torch.testing.assert_close(mass, area.new_tensor(float(len(events))))
        raw = mass - len(events)*result.intercept-ge.sum()
        torch.testing.assert_close(raw, result.objective+len(events)-len(events)*math.log(len(events)))
        torch.testing.assert_close(model.local_coefficients(q)-model.beta,model.h(q)[:,None]*model.u)
        self.assertEqual(float(model.h(model.reference).detach().abs().max()),0.)
        singular = torch.linalg.svdvals(model.local_coefficients(q)-model.beta)
        # Subtraction of beta loses relative accuracy when h is nearly zero.
        bound = 64 * torch.finfo(q.dtype).eps * model.local_coefficients(q).norm()
        self.assertLessEqual(float(singular[1].detach()),float(bound.detach()))
        # Independently evaluate refinement's fixed grid normalization comparison.
        ref = refinement(model)
        with torch.no_grad():
            q48,a48=quadrature(48)
            z24=normalized_density(model.g(q),area)[1]
            z48=normalized_density(model.g(q48),a48)[1]
        self.assertAlmostEqual(ref['comparisons'][0]['delta_log_z'],float(z48-z24),12)

    def test_no_bce_pseudo_negative_or_classification_sigmoid(self):
        events=sample_events(self.teacher,44,11)
        initial=copy.deepcopy(ReducedRankOneSyntheticPredictor(1011).state_dict())
        observed=[]
        def audited_fit(model, actual_events, q, area, **kwargs):
            self.assertTrue(torch.equal(actual_events,events))
            expected_q,expected_area=quadrature(24)
            self.assertTrue(torch.equal(q,expected_q))
            self.assertTrue(torch.equal(area,expected_area))
            observed.append(len(actual_events))
            return fit_rank_one_synthetic(model,actual_events,q,area,**kwargs)
        with synthetic_io_only(), \
             patch('torch.nn.functional.binary_cross_entropy',side_effect=AssertionError('BCE forbidden')), \
             patch('torch.nn.functional.binary_cross_entropy_with_logits',side_effect=AssertionError('BCE forbidden')), \
             patch('acawlr_ppp_teacher_student.fit_rank_one_synthetic',side_effect=audited_fit):
            model,_,_=fit_student(events,initial,steps=1)
        self.assertEqual(observed,[44])
        self.assertNotIn('teacher',inspect.signature(fit_student).parameters)
        with torch.no_grad():
            model.u.zero_(); model.beta.fill_(100.)
            xy=torch.tensor([[60000.,40000.],[-60000.,-40000.]],dtype=torch.float64)
            torch.testing.assert_close(model.g(xy),torch.tensor([200.,-200.],dtype=torch.float64))


class TeacherStudentFullExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads,old_threads)
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        with synthetic_io_only():
            cls.report,cls.archive=run_experiments()
        output=os.environ.get('ACAWLR_P1C_OUTPUT')
        if output:
            save_report(cls.report,cls.archive,output)

    def test_A_n256_recovery(self):
        failures=self.report['summaries']['A']['256']['recovery_failures']
        self.assertEqual(failures,[], '\n'.join(failures))

    def test_B_n256_recovery(self):
        failures=self.report['summaries']['B']['256']['recovery_failures']
        self.assertEqual(failures,[], '\n'.join(failures))

    def test_all_cases_preserved_and_constraints(self):
        self.assertEqual(len(self.report['runs']),20)
        self.assertEqual(len(self.archive['cases']),20)
        for case in self.report['runs']:
            with self.subTest(case=case['id']):
                saved=self.archive['cases'][case['id']]
                self.assertEqual(state_hash(saved['initial_state']),case['initial_state_sha256'])
                self.assertEqual(state_hash({'events':saved['events']}),case['event_sha256'])
                self.assertEqual(len(saved['history']),301)
                self.assertTrue(case['final_gradient']['finite'])
                self.assertEqual(case['reference_error'],0.)
                self.assertLessEqual(case['rank_one_absolute_residual'],case['rank_one_roundoff_bound'])
                self.assertTrue(all(math.isfinite(case[k]) for k in ('intensity_rmse','g_rmse','coefficient_rmse')))
        # Shared A/B anchor is independently trained twice with bit-identical inputs.
        for n in (44,256):
            a=self.archive['cases'][f'A_n{n}_data11_init1011']
            b=self.archive['cases'][f'B_n{n}_data11_init1011']
            self.assertEqual(state_hash(a['final_state']),state_hash(b['final_state']))

    def test_student_quadrature_refinement(self):
        for case in self.report['runs']:
            for comparison in case['quadrature']['comparisons']:
                with self.subTest(case=case['id'],coarse=comparison['coarse']):
                    self.assertTrue(comparison['stable'],str(comparison))


if __name__=='__main__':
    unittest.main()
