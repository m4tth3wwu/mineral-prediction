"""Scientific interpretation guards; synthetic scalar records only, no model updates."""
import unittest
from report_acawlr_ppp_p1g_empirical import conflict,flags
class InterpretationTests(unittest.TestCase):
    def fixture(self,kind):
        h=[];obs=[]
        for i in range(4):
            total=100.-i;data=10.+i if kind=='conflict' else 10.-i
            if kind=='unpenalized':data=total
            metric=dict(KL=.01+i*.01,centered_g_area_RMSE=.1+i*.1,q_ratio=1.+i*.1,field_ratio=1.+i*.1)
            h.append(dict(step=i*25,total_objective=total,empirical_ppp=data,penalty_total=total-data,**metric))
            obs.append(dict(step=i*25,total_objective=total,empirical_ppp=data,penalty_total=total-data,**{'96':metric}))
        # fixed-interval report expects the regular 0..2000 monitor timeline.
        for step in range(100,2001,25):
            h.append(dict(h[-1],step=step));obs.append(dict(obs[-1],step=step))
        return h,obs
    def test_three_independent_intervals_required(self):
        h,o=self.fixture('conflict');r=conflict(h,o);self.assertTrue(r['sustained_conflict']);self.assertEqual(r['longest_monitor_conflict_streak'],3)
    def test_zero_regularization_cannot_mask_data_degradation(self):
        h,o=self.fixture('unpenalized');r=conflict(h,o);self.assertEqual(r['step48_conflicts'],[]);self.assertFalse(r['sustained_conflict'])
    def test_generalization_tradeoff_is_not_regularizer_conflict(self):
        h,o=self.fixture('overfit');r=conflict(h,o);self.assertEqual(r['monitor96_conflicts'],[]);self.assertEqual(len(r['overfit_monitor_intervals']),3)
    def test_single_interval_not_sustained(self):
        h,o=self.fixture('conflict')
        for v in h[2:]:v['empirical_ppp']=9.
        for v in o[2:]:v['empirical_ppp']=9.
        self.assertFalse(conflict(h,o)['sustained_conflict'])
if __name__=='__main__':unittest.main()
