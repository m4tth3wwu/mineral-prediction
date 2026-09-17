"""Read-only LR experiment verification: no extra fits or new data."""
import json
import math
from pathlib import Path
import unittest
import torch
from acawlr_ppp_bridge import ReducedRankOneSyntheticPredictor, rank_one_value_and_grad
from acawlr_ppp_synthetic import PENALTY, quadrature, evaluate
from acawlr_ppp_optimization import Trajectory, CHECKPOINTS, sha256
from acawlr_ppp_p1d_lr import STEM, RATES, SEEDS, LearningRateTrajectory, vector


class LearningRateDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads,old_threads)
        cls.r=json.loads((cls.root/(STEM+'_RESULTS.json')).read_text())
        cls.a=torch.load(cls.root/(STEM+'_STATES.pt'),weights_only=True)
        cls.p1c=torch.load(cls.root/'ACAWLR_PPP_P1C_TS_STATES.pt',weights_only=True)
        cls.baseline=json.loads((cls.root/'ACAWLR_PPP_P1D_OPTIMIZATION_RESULTS.json').read_text())
        cls.teacher=ReducedRankOneSyntheticPredictor(0)
        cls.teacher.load_state_dict(cls.a['teacher_state']);cls.teacher.requires_grad_(False)

    def test_exact_inputs_and_only_requested_conditions(self):
        self.assertEqual(self.r['status'],'complete')
        self.assertEqual({(c['learning_rate'],c['data_seed']) for c in self.r['cases']},
                         {(lr,s) for lr in RATES for s in SEEDS})
        self.assertEqual(len(self.r['cases']),9)
        for name,value in self.a['teacher_state'].items():
            self.assertTrue(torch.equal(value,self.p1c['teacher_state'][name]))
        for c in self.r['cases']:
            saved=self.a['cases'][f"{c['learning_rate']}_{c['data_seed']}"]
            old=self.p1c['cases'][f"B_n256_data{c['data_seed']}_init1011"]
            self.assertEqual(c['n'],256);self.assertEqual(c['init_seed'],1011)
            self.assertTrue(torch.equal(saved['events'],old['events']))
            for name,value in saved['initial_state'].items():
                self.assertTrue(torch.equal(value,old['initial_state'][name]))
                self.assertTrue(torch.equal(value,saved['checkpoints'][0]['model_state'][name]))

    def test_same_training_loop_optimizer_settings_and_checkpoint_schedule(self):
        self.assertIs(LearningRateTrajectory.advance,Trajectory.advance)
        old_groups=torch.load(self.root/'ACAWLR_PPP_P1D_OPTIMIZATION_STATES.pt',weights_only=True)['cases'][23]['checkpoints'][2000]['optimizer_state']['param_groups']
        for c in self.r['cases']:
            self.assertEqual(c['last_step'],2000)
            self.assertEqual([h['step'] for h in c['history']],list(range(2001)))
            self.assertEqual([h['step'] for h in c['observations']],list(range(0,2001,25)))
            cps=self.a['cases'][f"{c['learning_rate']}_{c['data_seed']}"]['checkpoints']
            self.assertEqual(set(cps),set(CHECKPOINTS))
            for step,cp in cps.items():
                group=cp['optimizer_state']['param_groups'][0]
                for k,v in old_groups[0].items():
                    self.assertEqual(group[k],c['learning_rate'] if k=='lr' else v)
                if step:
                    self.assertTrue(all(int(s['step'])==step for s in cp['optimizer_state']['state'].values()))
        protocol=self.r['protocol']['unchanged_baseline_protocol']
        self.assertEqual(protocol['penalty'],dict(beta=.02,u=.2,theta=.001))
        self.assertEqual(protocol['training_quadrature'],24);self.assertEqual(protocol['evaluation_quadrature'],48)

    def test_supplemented_baseline_exactly_reproduces_history(self):
        for c in self.r['cases']:
            if c['learning_rate']!=.01: continue
            self.assertTrue(c['baseline_replay']['passed'])
            self.assertTrue(c['baseline_replay']['all_model_optimizer_checkpoints_bit_identical'])
            old=next(o for o in self.baseline['cases'] if o['data_seed']==c['data_seed'])
            for a,b in zip(c['history'],old['history']):
                self.assertEqual(a['ppp_objective'],b['profile_objective'])
                self.assertEqual(a['total_loss'],b['objective'])
                self.assertEqual(a['gradient_l2'],b['gradient_l2'])

    def test_loss_components_and_actual_parameter_updates(self):
        for c in self.r['cases']:
            for h in c['history']:
                self.assertNotIn('objective',h)
                self.assertAlmostEqual(h['total_loss'],h['ppp_objective']+h['regularization_total'],places=10)
                self.assertAlmostEqual(h['regularization_total'],sum(h['regularization_components'].values()),places=10)
                self.assertTrue(math.isfinite(h['relative_update_norm']))
                self.assertFalse(h['nan_inf'])
            cps=self.a['cases'][f"{c['learning_rate']}_{c['data_seed']}"]['checkpoints']
            for step,cp in cps.items():
                row=c['history'][step]
                previous=cp['previous_parameter_vector'];now=cp['parameter_vector']
                self.assertAlmostEqual(float((now-previous).norm()),row['parameter_update_l2'],places=12)
                self.assertAlmostEqual(float((now-previous).norm())/max(float(previous.norm()),1e-30),row['relative_update_norm'],places=12)
                model=ReducedRankOneSyntheticPredictor(0);model.load_state_dict(cp['model_state'])
                self.assertTrue(torch.equal(now,vector(model)))
                self.assertEqual(sum(p.numel() for p in model.parameters()),675)

    def test_checkpoint_ppp_gradient_and_dense_predictor_recovery(self):
        q,area=quadrature(24);grid,weights=quadrature(48)
        for c in self.r['cases']:
            saved=self.a['cases'][f"{c['learning_rate']}_{c['data_seed']}"]
            for step in (300,2000):
                with self.subTest(lr=c['learning_rate'],seed=c['data_seed'],step=step):
                    cp=saved['checkpoints'][step]
                    model=ReducedRankOneSyntheticPredictor(0);model.load_state_dict(cp['model_state'])
                    row=next(o for o in c['observations'] if o['step']==step)
                    result,reg,loss,grads=rank_one_value_and_grad(model,saved['events'],q,area,PENALTY)
                    direct=256*torch.logsumexp(area.log()+model.g(q),0)-model.g(saved['events']).sum()
                    torch.testing.assert_close(result.objective,direct,rtol=1e-12,atol=1e-10)
                    self.assertAlmostEqual(float(loss),row['total_loss'],places=9)
                    self.assertAlmostEqual(float(result.objective.detach()),row['ppp_objective'],places=9)
                    self.assertAlmostEqual(float(sum(g.square().sum() for g in grads).sqrt()),row['gradient_l2'],places=8)
                    with torch.no_grad():
                        torch.testing.assert_close(model.g(grid),cp['g_surface'],rtol=0,atol=0)
                        torch.testing.assert_close(model.local_coefficients(grid),cp['coefficient_surface'],rtol=0,atol=0)
                    metrics,_=evaluate(model,self.teacher,grid,weights)
                    for k in ('intensity_correlation','intensity_rmse','g_correlation','g_rmse','coefficient_rmse'):
                        self.assertAlmostEqual(metrics[k],row[k],places=10)
        with torch.no_grad():
            model.u.zero_();model.beta.fill_(100.)
            xy=grid.new_tensor([[60000.,40000.],[-60000.,-40000.]])
            torch.testing.assert_close(model.g(xy),grid.new_tensor([200.,-200.]),rtol=0,atol=0)

    def test_history_and_instrumentation_source_unchanged(self):
        self.assertTrue(self.r['historical_files_unchanged'])
        for path,digest in self.r['protocol']['historical_sha256'].items():
            self.assertEqual(sha256(self.root.parent/path),digest,path)
        self.assertEqual(sha256(self.root/'acawlr_ppp_p1d_lr.py'),self.r['protocol']['runner_sha256'])


if __name__=='__main__': unittest.main()
