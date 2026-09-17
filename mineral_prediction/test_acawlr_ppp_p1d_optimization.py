"""Read-only verification of the three P1-D trajectories; never train extra fits."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from acawlr_ppp_bridge import ReducedRankOneSyntheticPredictor, rank_one_value_and_grad
from acawlr_ppp_synthetic import PENALTY, quadrature, evaluate
from acawlr_ppp_teacher_student import state_hash
from acawlr_ppp_optimization import STEM, SEEDS, CHECKPOINTS, history_manifest


class OptimizationDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent
        cls.old_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads,cls.old_threads)
        cls.report=json.loads((cls.root/(STEM+'_RESULTS.json')).read_text())
        cls.states=torch.load(cls.root/(STEM+'_STATES.pt'),weights_only=True)
        cls.old_report=json.loads((cls.root/'ACAWLR_PPP_P1C_TS_RESULTS.json').read_text())
        cls.old_states=torch.load(cls.root/'ACAWLR_PPP_P1C_TS_STATES.pt',weights_only=True)
        cls.teacher=ReducedRankOneSyntheticPredictor(0)
        cls.teacher.load_state_dict(cls.states['teacher_state'])
        cls.teacher.requires_grad_(False)

    def test_exact_archived_inputs_only_three_n256_cases(self):
        self.assertEqual([c['data_seed'] for c in self.report['cases']],list(SEEDS))
        self.assertEqual(set(self.states['cases']),set(SEEDS))
        for key,value in self.states['teacher_state'].items():
            self.assertTrue(torch.equal(value,self.old_states['teacher_state'][key]))
        for c in self.report['cases']:
            saved=self.states['cases'][c['data_seed']]
            old=self.old_states['cases'][f"B_n256_data{c['data_seed']}_init1011"]
            self.assertEqual(c['n'],256)
            self.assertEqual(saved['events'].shape,(256,2))
            self.assertTrue(torch.equal(saved['events'],old['events']))
            for key,value in saved['initial_state'].items():
                self.assertTrue(torch.equal(value,old['initial_state'][key]))
                self.assertTrue(torch.equal(value,saved['checkpoints'][0]['model_state'][key]))
            self.assertEqual(c['event_sha256'],state_hash({'events':saved['events']}))
            self.assertEqual(c['initial_state_sha256'],state_hash(saved['initial_state']))

    def test_step300_reproduces_p1c_and_first301_objectives(self):
        for c in self.report['cases']:
            with self.subTest(seed=c['data_seed']):
                self.assertTrue(c['reproduction']['passed'])
                self.assertTrue(c['reproduction']['model_state_bit_identical'])
                self.assertEqual(c['reproduction']['history_max_abs_error'],0.)
                old=next(o for o in self.old_report['runs'] if o['id']==f"B_n256_data{c['data_seed']}_init1011")
                now=next(o for o in c['observations'] if o['step']==300)
                for key in ('intensity_correlation','intensity_rmse','g_correlation','h_std'):
                    self.assertAlmostEqual(now[key],old[key],places=10)
                self.assertAlmostEqual(now['objective'],old['final']['objective'],places=10)
                self.assertAlmostEqual(now['gradient_l2'],old['final_gradient']['l2'],places=10)

    def test_optimizer_continuity_and_unchanged_configuration(self):
        p=self.report['protocol']
        self.assertEqual(p['horizon'],2000)
        self.assertFalse(p['early_stopping'])
        self.assertEqual(p['penalty'],dict(beta=.02,u=.2,theta=.001))
        self.assertEqual(p['training_quadrature'],24)
        self.assertEqual(p['evaluation_quadrature'],48)
        defaults=p['optimizer_defaults']
        self.assertEqual(defaults['lr'],.01)
        self.assertEqual(defaults['betas'],[.9,.999])
        self.assertEqual(defaults['eps'],1e-8)
        self.assertEqual(defaults['weight_decay'],0)
        self.assertFalse(defaults['amsgrad'])
        model=ReducedRankOneSyntheticPredictor(0)
        self.assertEqual(sum(v.numel() for v in model.parameters()),675)
        for saved in self.states['cases'].values():
            self.assertEqual(set(saved['checkpoints']),set(CHECKPOINTS))
            for step,cp in saved['checkpoints'].items():
                self.assertEqual(set(cp['model_state']),set(model.state_dict()))
                for key,v in cp['model_state'].items():
                    self.assertEqual(v.shape,model.state_dict()[key].shape)
                opt=cp['optimizer_state']
                self.assertEqual(opt['param_groups'][0]['lr'],.01)
                if step==0:
                    self.assertEqual(opt['state'],{})
                else:
                    self.assertEqual(len(opt['state']),len(list(model.parameters())))
                    self.assertTrue(all(int(s['step'])==step for s in opt['state'].values()))
                    self.assertTrue(all('exp_avg' in s and 'exp_avg_sq' in s for s in opt['state'].values()))

    def test_complete_unselected_trajectories_and_finite_checks(self):
        self.assertEqual(self.report['status'],'complete')
        for c in self.report['cases']:
            self.assertEqual(c['last_step'],2000)
            self.assertEqual([h['step'] for h in c['history']],list(range(2001)))
            self.assertEqual([h['step'] for h in c['observations']],list(range(0,2001,25)))
            for h in c['history']:
                self.assertAlmostEqual(h['objective'],h['profile_objective']+h['penalty'],places=10)
                self.assertAlmostEqual(h['gradient_l2_per_n'],h['gradient_l2']/256,places=12)
                self.assertFalse(h['nan_inf'])
            for h in c['observations']:
                for key in ('g_rmse','coefficient_rmse','top20_overlap','h_std','u_norm','spatial_contribution_std','peak_relative_intensity'):
                    self.assertTrue(math.isfinite(h[key]))

    def test_checkpoint_metrics_objective_and_no_forbidden_loss(self):
        q,area=quadrature(24)
        grid,weights=quadrature(48)
        for case in self.report['cases']:
            saved=self.states['cases'][case['data_seed']]
            for step in CHECKPOINTS:
                with self.subTest(seed=case['data_seed'],step=step):
                    model=ReducedRankOneSyntheticPredictor(0)
                    model.load_state_dict(saved['checkpoints'][step]['model_state'])
                    expected=next(o for o in case['observations'] if o['step']==step)
                    with patch('torch.nn.functional.binary_cross_entropy',side_effect=AssertionError('BCE')), \
                         patch('torch.nn.functional.binary_cross_entropy_with_logits',side_effect=AssertionError('BCE')):
                        result,penalty,total,grads=rank_one_value_and_grad(model,saved['events'],q,area,PENALTY)
                    # Independent expression has only true events and quadrature integral.
                    direct=256*torch.logsumexp(area.log()+model.g(q),0)-model.g(saved['events']).sum()
                    torch.testing.assert_close(result.objective,direct,rtol=1e-12,atol=1e-10)
                    self.assertAlmostEqual(float(total),expected['objective'],places=9)
                    self.assertAlmostEqual(float(sum(g.square().sum() for g in grads).sqrt()),expected['gradient_l2'],places=8)
                    metrics,surface=evaluate(model,self.teacher,grid,weights)
                    for key in ('intensity_correlation','intensity_rmse','g_correlation','g_rmse','coefficient_rmse','top20_overlap'):
                        self.assertAlmostEqual(metrics[key],expected[key],places=10)
                    torch.testing.assert_close(surface,saved['checkpoints'][step]['relative_intensity'],rtol=0,atol=0)
                    with torch.no_grad():
                        h=model.h(grid); mean=(h*weights/weights.sum()).sum()
                        std=float(((h-mean).square()*weights/weights.sum()).sum().sqrt())
                    self.assertAlmostEqual(std,expected['h_std'],places=12)
        # The exact reused model has no final classification sigmoid.
        with torch.no_grad():
            model.u.zero_(); model.beta.fill_(100.)
            xy=torch.tensor([[60000.,40000.],[-60000.,-40000.]],dtype=torch.float64)
            torch.testing.assert_close(model.g(xy),xy.new_tensor([200.,-200.]),rtol=0,atol=0)

    def test_all_historical_files_unchanged(self):
        self.assertTrue(self.report['historical_files_unchanged'])
        self.assertEqual(history_manifest(self.root),self.report['protocol']['historical_sha256'])


if __name__=='__main__':
    unittest.main()
