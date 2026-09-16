"""Independent consistency checks on the delivered real-data experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import PPP_binary_lithology_v3 as v3

out=v3.ROOT/'output/ppp_binary_lithology_v3'
cache=v3.ROOT/'output/ppp_binary_features_v3'
fine,events,external=v3.load_features(v3.ROOT/'output/ppp_lithology_spatial_validation_5km',cache,v3.legacy.Config())
assert len(fine)==132888 and len(events)==44 and len(external)==16
assert set(fine.lith_intrusive.unique())<={0,1}
assert events.lithology_valid.all() and external.lithology_valid.all()
assert np.isfinite(fine[[v3.NEW_DISTANCE,'dist_binary_contact_km']]).all().all()
assert (fine.dist_binary_contact_km+1e-8>=fine.dist_intrusive_contact_km).all()
held=pd.read_csv(out/'held_out_event_predictions.csv')
assert len(held)==5*3*44
assert (held.groupby(['model','seed']).deposit.nunique()==44).all()
compact=pd.read_csv(out/'compact_held_out_events.csv')
assert len(compact)==5*44
assert (compact.groupby('model').deposit.nunique()==44).all()
for name in ['compact_fold_metrics.csv','spatial_fold_metrics.csv']:
    frame=pd.read_csv(out/name)
    assert (frame.minimum_event_separation_km>50).all()
    assert np.isfinite(frame.presence_background_auc).all()
influence=pd.read_csv(out/'event_influence.csv')
assert len(influence)==44 and influence.deposit.nunique()==44
stable=pd.read_csv(out/'target_stability.csv')
assert len(stable)==len(fine)
assert stable.top10_selection_frequency.between(0,1).all()
assert np.isfinite(stable.normalized_log_intensity_sd).all()
residuals=pd.read_csv(out/'regional_count_residuals.csv')
assert residuals.observed.sum()==44
assert abs(residuals.predicted.sum()-44)<1e-6
scores=pd.read_csv(out/'external_deposit_scores.csv')
assert len(scores)==80 and (scores.groupby('model').DEPOSIT.nunique()==16).all()
pred=pd.read_csv(out/'predictions_5km.csv')
area=pred.area_km2.to_numpy()
old=pred.M2_old_polygon_contact_intensity.to_numpy()
new=pred.M2_binary_contact_intensity.to_numpy()
a=old>=v3.ppp.weighted_quantile(old,area,.9)
b=new>=v3.ppp.weighted_quantile(new,area,.9)
summary={'checks':'passed','feature_cache':'source and output fingerprints verified; reused',
         'training_events':44,'external_events':16,'spatial_models':5,
         'old_new_map_spearman':float(spearmanr(old,new).statistic),
         'old_new_top10_area_jaccard':float(area[a&b].sum()/area[a|b].sum()),
         'stability_refits':len(pd.read_csv(out/'stability_refits.csv')),
         'single_event_min_top10_jaccard':float(influence.top10_area_jaccard.min()),
         'invalid_lithology_support_cells':int((~fine.lithology_valid).sum())}
v3.save_json(out/'verification.json',summary)
print(json.dumps(summary,indent=2))
