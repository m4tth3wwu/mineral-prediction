"""P1-F tests: gradients, target isolation, classification and dose gating; no updates."""
import copy,inspect,unittest
import torch
import acawlr_ppp_p1f_population as f
class PopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.teacher,cls.starts,cls.direction=f.original_inputs();cls.xy,cls.w=f.e.quadrature(4);cls.t=f.e.target(cls.teacher.g(cls.xy),cls.w)
    def test_same_initializations_and_buffers(self):
        self.assertEqual(f.e.state_hash(self.starts['standard']),f.e.INIT_HASH)
        again=f.original_inputs()[1];self.assertEqual(f.e.state_hash(again['local']),f.e.state_hash(self.starts['local']))
        local=f.e.model(self.starts['local']);v=f.e.vec(self.teacher);self.assertAlmostEqual(float((f.e.vec(local)-v).norm()),1e-3*max(1,float(v.norm())),places=12)
        for n,b in self.teacher.named_buffers():torch.testing.assert_close(b,dict(local.named_buffers())[n],rtol=0,atol=0)
    def test_population_teacher_gradient_and_target_detach(self):
        m=f.e.model(self.teacher.state_dict());grad=f.e.flat_grad(f.e.pop(m.g(self.xy),self.t),m)
        self.assertLess(float(grad.abs().max()),1e-9);self.assertFalse(self.t['pi'].requires_grad)
    def test_matched_gradient_difference_exact_penalty(self):
        m=f.e.model(self.starts['local']);g0=f.e.flat_grad(256*f.e.pop(m.g(self.xy),self.t),m);g1=f.e.flat_grad(256*f.e.pop(m.g(self.xy),self.t)+f.e.PENALTY(m),m);gp=f.e.flat_grad(f.e.PENALTY(m),m)
        torch.testing.assert_close(g1-g0,gp,rtol=1e-10,atol=1e-10)
        gs=f.e.flat_grad(256*f.e.pop(m.g(self.xy),self.t)+.1*f.e.PENALTY(m),m);torch.testing.assert_close(gs-g0,.1*gp,rtol=1e-10,atol=1e-10)
    def test_forward_formula_identity(self):
        m=f.e.model(self.starts['standard']);h=m.h(self.xy);g=(self.xy/f.e.SCALE*(m.beta+h[:,None]*m.u)).sum(1)
        torch.testing.assert_close(g,m.g(self.xy),rtol=0,atol=0)
    def test_kl_identity_and_weight_perturbation(self):
        g=self.t['g']+.2*torch.sin(self.xy[:,0]);self.assertAlmostEqual(float(f.e.pop(g,self.t)-f.e.pop(self.t['g'],self.t)),f.e.measures(g,self.t)['KL'],places=12)
        with self.assertRaises(ValueError):f.e.target(g,-self.w)
    def test_collapse_is_function_level_not_u_norm(self):
        scale=dict(teacher_q_RMS=.15,teacher_field_RMS=.4)
        v=dict(q=dict(RMS=1e-13,max_abs=1e-12),field_contribution_RMS=1e-13,q_ratio=1e-12,field_ratio=1e-12,KL=.006,TV=.04,centered_g_area_RMSE=.1,centered_g_teacher_RMSE=.1,u_norm=100)
        self.assertTrue(f.labels(v,scale,.006)['collapse'])
        v['field_contribution_RMS']=.1;self.assertFalse(f.labels(v,scale,.006)['collapse'])
    def test_mild_and_severe_distinct(self):
        scale=dict(teacher_q_RMS=.15,teacher_field_RMS=.4)
        v=dict(q=dict(RMS=.06,max_abs=.1),field_contribution_RMS=.2,q_ratio=.4,field_ratio=.5,KL=1e-5,TV=.001,centered_g_area_RMSE=.001,centered_g_teacher_RMSE=.001)
        self.assertEqual(f.labels(v,scale,.006)['category'],'mild_shrinkage');v['KL']=.001;self.assertEqual(f.labels(v,scale,.006)['category'],'severe_shrinkage')
    def test_dose_requires_complete_matched_fidelity_and_effect(self):
        base=dict(status='COMPLETED',steps=2000,fidelity=dict(passed=True),density_success=True,predictor_success=True,stable_severe_shrinkage=False,stable_collapse=False)
        r={n:copy.deepcopy(base) for n in ['standard_P0','standard_Plambda','local_P0','local_Plambda']}
        self.assertFalse(f.dose_trigger(r)['eligible']);r['local_Plambda'].update(density_success=False,stable_severe_shrinkage=True);self.assertTrue(f.dose_trigger(r)['eligible'])
        r['standard_Plambda']['fidelity']['passed']=False;self.assertFalse(f.dose_trigger(r)['eligible'])
        r['standard_Plambda']['fidelity']['passed']=True;r['local_P0']['predictor_success']=False;self.assertFalse(f.dose_trigger(r)['eligible'])
    def test_legacy_failure_cannot_enter_fresh_gate(self):
        source=inspect.getsource(f.prepare);self.assertIn('blocks_fresh=False',source);self.assertNotIn("checks['legacy_fidelity']",source)
        self.assertEqual(f.read(f.OLD/'POPULATION_RESULTS.json')['count'],0)
    def test_historical_frozen_integrity(self):self.assertTrue(f.frozen_check()['passed'])
if __name__=='__main__':unittest.main()
