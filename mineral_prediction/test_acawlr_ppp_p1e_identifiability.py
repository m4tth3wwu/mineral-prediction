"""P1-E no-training tests only; never execute legacy suites or trajectory loops."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import acawlr_ppp_p1e_identifiability as p
class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.archive=torch.load(p.ROOT/'ACAWLR_PPP_P1C_TS_STATES.pt',weights_only=True,map_location='cpu')
        cls.teacher=p.model(cls.archive['teacher_state']);cls.xy,cls.w=p.quadrature(4)
    def test_formal_hash_and_buffers(self):
        self.assertEqual(p.state_hash(self.teacher.state_dict()),p.TEACHER_HASH)
        self.assertEqual(p.state_hash(self.archive['cases']['B_n256_data23_init1011']['initial_state']),p.INIT_HASH)
        self.assertEqual(set(dict(self.teacher.named_buffers())),{'anchors','reference','angle'})
    def test_detached_targets_and_invalid_weights(self):
        g=self.teacher.g(self.xy);t=p.target(g,self.w)
        self.assertFalse(t['pi'].requires_grad);self.assertAlmostEqual(float(t['pi'].sum()),1,places=14)
        with self.assertRaises(ValueError):p.target(g,-self.w)
        with self.assertRaises(ValueError):p.target(g,self.w[:-1])
    def test_kl_identity_and_constant_gauge(self):
        t=p.target(self.teacher.g(self.xy),self.w);g=t['g']+.3*torch.sin(self.xy[:,0])
        self.assertAlmostEqual(float(p.pop(g,t)-p.pop(t['g'],t)),p.measures(g,t)['KL'],places=13)
        self.assertAlmostEqual(p.measures(g,t)['KL'],p.measures(g+3,t)['KL'],places=13)
    def test_normalizer_limits_and_derivatives(self):
        z,m,v=p.global_functions(np.array([0.,1e-8,1e-3,.049,.051,1.,-5.,1000.]))
        self.assertTrue(np.isfinite(z).all());self.assertTrue((v>0).all());self.assertEqual(z[0],0)
        for b in (.001,.049,.051,.5,5.,-10.):
            eps=1e-5
            self.assertAlmostEqual(float((p.global_functions(b+eps)[0]-p.global_functions(b-eps)[0])/(2*eps)),float(p.global_functions(b)[1]),places=8)
    def test_global_inverse_moments(self):
        expected=np.array([.5,-.3]);actual=p.solve_global(p.global_functions(expected)[1],0)
        np.testing.assert_allclose(actual['beta'],expected,atol=1e-12);self.assertTrue(actual['converged'])
    def test_normalizer_dense_uniform(self):
        b=torch.tensor([.3,-.2],dtype=p.DT);xy,w=p.quadrature(192)
        approx=float(torch.logsumexp(w.log()+xy/p.SCALE@b,0));exact=np.log(9600)+p.global_functions(b.numpy())[0].sum()
        self.assertLess(abs(approx-exact),1e-6)
    def test_fixed_h_solver_oracle(self):
        s=p.surface(self.teacher,self.xy);t=p.target(s['g'],self.w);X=self.xy/p.SCALE;D=torch.cat([X,s['h'][:,None]*X],1)
        fit=p.convex_newton(D,t,[0]*4);self.assertTrue(fit['converged']);self.assertLess(abs(fit['KL']),1e-10)
    def test_penalty_scaling_and_all_parameters(self):
        m=p.model(self.teacher.state_dict());a=p.PENALTY(m);parts=p.penalty_parts(m)
        self.assertAlmostEqual(float(a.detach()),sum(parts.values()),places=13)
        self.assertAlmostEqual(parts['u'],4.5,places=13)
        grad=p.flat_grad(a,m);expected=torch.cat([(.02 if n=='beta' else .2 if n=='u' else .001)*v.detach().flatten() for n,v in m.named_parameters()])
        torch.testing.assert_close(grad,expected,atol=1e-14,rtol=1e-14)
    def test_symmetry_and_anchor(self):
        m=p.model(self.teacher.state_dict())
        with torch.no_grad():m.u.neg_();m.cawnn.readout.weight.neg_()
        torch.testing.assert_close(m.g(self.xy),self.teacher.g(self.xy),rtol=1e-13,atol=1e-13)
        self.assertEqual(float(m.h(m.reference).detach()),0.)
        self.assertAlmostEqual(float(p.PENALTY(m).detach()),float(p.PENALTY(self.teacher).detach()),places=13)
    def test_canonical_undefined_zero_branch(self):
        c=p.canonical(self.teacher,torch.zeros(16,dtype=p.DT),self.w)
        self.assertFalse(c['defined']);self.assertIsNone(c['v'])
    def test_chain_jacobian_finite_difference(self):
        m=p.model(self.teacher.state_dict());xy=self.xy[:3];_,jg,jb=p.jacobians(m,xy);base=p.vec(m)
        d=torch.randn(base.shape,generator=torch.Generator().manual_seed(17),dtype=p.DT);d/=d.norm();eps=1e-6
        p.setvec(m,base+eps*d);a=p.surface(m,xy);p.setvec(m,base-eps*d);b=p.surface(m,xy)
        torch.testing.assert_close((a['g']-b['g'])/(2*eps),jg@d,atol=1e-7,rtol=1e-4)
        torch.testing.assert_close(((a['b']-b['b'])/(2*eps)).flatten(),jb@d,atol=1e-7,rtol=1e-4)
    def test_output_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):p.reserve_output(Path(d))
    def test_frozen_integrity(self):
        active=p.read(p.ROOT/'docs/project_memory/ACTIVE_SESSION.json');manifest=p.read(p.ROOT/active['frozen_manifest'])
        for name,h in manifest.items():self.assertEqual(p.sha(p.ROOT/name),h,name)
    def test_memory_event_schema_and_links(self):
        memory=p.ROOT/'docs/project_memory';rows=[json.loads(x) for x in (memory/'RUN_INDEX.jsonl').read_text(encoding='utf-8').splitlines()]
        ids=set();runs={}
        for row in rows:
            self.assertNotIn(row['event_id'],ids);ids.add(row['event_id'])
            self.assertIn(row['scientific_status'],('passed','not_met','not_evaluated','numerical_unresolved'))
            self.assertNotEqual(row['status'],row['scientific_status'])
            for key in ('protocol_path','provenance_path','source_manifest_path'):
                if row[key] is not None:self.assertTrue((p.ROOT/row[key]).exists())
            previous=runs.setdefault(row['run_id'],row['protocol_sha256']);self.assertEqual(previous,row['protocol_sha256'])
        # Production writer must append rather than replace the historical stream.
        import inspect
        self.assertIn(".open('a',encoding='utf-8')",inspect.getsource(p.event))
if __name__=='__main__':unittest.main()
