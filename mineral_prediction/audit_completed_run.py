"""Audit saved run, using independent geometric checks and one fold replay."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from pyproj import Transformer
from threadpoolctl import threadpool_limits
import PPP_three_papers_simple as ppp
import PPP_binary_lithology_v3 as model
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'output/ppp_run_20260905_195450_443'
OUT=RUN/'verification'
CACHE=ROOT/'output/ppp_binary_features_v3'
def save(df,name):df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 fine=pd.read_csv(CACHE/'grid.csv'); events=pd.read_csv(CACHE/'events.csv'); ext=pd.read_csv(CACHE/'external.csv')
 manifest=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
 hashes=[]
 for path,expected in manifest['signature']['files'].items():
  ok=digest(Path(path))==expected;hashes.append(dict(file=path,hash_matches=ok));assert ok,path
 for name,expected in manifest['outputs'].items():
  ok=digest(CACHE/name)==expected;hashes.append(dict(file=str(CACHE/name),hash_matches=ok));assert ok,name
 conf=json.loads((RUN/'config.json').read_text(encoding='utf-8'));assert digest(CACHE/'manifest.json')==conf['feature_manifest_sha256']
 save(pd.DataFrame(hashes),'cache_provenance.csv')
 source=pd.read_csv(ROOT/'output/paper_point_matching/paper_figure2g_all_60_points_model_ready.csv').set_index('paper_star_id')
 used=pd.concat([events.assign(split='training'),ext.assign(split='validation')],ignore_index=True)
 s=source.loc[used.paper_star_id]
 delta=np.abs(used[['longitude','latitude']].to_numpy()-s[['LONGITUDE','LATITUDE']].to_numpy())
 assert delta.max()<1e-9
 assert list(used.split)==list(s.paper_split)
 trans=Transformer.from_crs('EPSG:4326','ESRI:102039',always_xy=True)
 x,y=trans.transform(used.longitude.to_numpy(),used.latitude.to_numpy())
 projection_error=np.hypot(x-used.metric_x,y-used.metric_y)
 assert projection_error.max()<0.01
 near=cKDTree(fine[['metric_x','metric_y']]).query(used[['metric_x','metric_y']])[0]/1000
 coords=used[['paper_star_id','DEPOSIT','split','longitude','latitude']].copy()
 coords['source_coordinate_max_error_deg']=delta.max(axis=1);coords['projection_error_m']=projection_error
 coords['nearest_integration_point_km']=near
 save(coords,'coordinate_comparison.csv')
 coarse=ppp.coarsen_quadrature(fine,10)
 assignments=pd.read_csv(RUN/'spatial_block_assignments.csv')
 held=pd.read_csv(RUN/'held_out_event_predictions.csv');held=held[held.model.eq('M2_binary_contact')]
 metrics=pd.read_csv(RUN/'spatial_fold_metrics.csv');metrics=metrics[metrics.model.eq('M2_binary_contact')]
 folds=[];roles=[];replay=None
 for seed,table in assignments.groupby('seed'):
  mapping={(int(r.block_x),int(r.block_y)):int(r.fold) for r in table.itertuples()}
  ef=np.array([mapping[k] for k in ppp.block_keys(events,200)])
  qf=np.array([mapping[k] for k in ppp.block_keys(coarse,200)])
  assert len(held[held.seed.eq(seed)])==44
  for fold in range(1,6):
   region=shapely.union_all([shapely.box(bx*200000,by*200000,(bx+1)*200000,(by+1)*200000) for (bx,by),v in mapping.items() if v==fold])
   de=shapely.distance(shapely.points(events.metric_x,events.metric_y),region)
   dq=shapely.distance(shapely.points(coarse.metric_x,coarse.metric_y),region)
   et=(ef!=fold)&(de>50000);qt=(qf!=fold)&(dq>50000);test=ef==fold
   names=set(events.loc[test,'DEPOSIT']);saved=held[held.seed.eq(seed)&held.fold.eq(fold)]
   assert names==set(saved.deposit)
   row=metrics[metrics.seed.eq(seed)&metrics.fold.eq(fold)].iloc[0]
   sep=cdist(events.loc[et,['metric_x','metric_y']],events.loc[test,['metric_x','metric_y']]).min()/1000
   assert sep>50 and np.isclose(sep,row.minimum_event_separation_km)
   assert et.sum()==row.train_events and qt.sum()==row.train_quadrature and test.sum()==row.test_events
   assert not set(events.loc[et,'DEPOSIT']) & names
   folds.append(dict(method='blocks',seed=seed,fold=fold,train=int(et.sum()),test=int(test.sum()),guard_excluded=int((~et&~test).sum()),minimum_distance_km=sep,passed=True))
   for i,r in events.iterrows():roles.append(dict(seed=seed,fold=fold,paper_star_id=r.paper_star_id,deposit=r.DEPOSIT,role='test' if test[i] else 'train' if et[i] else 'guard_excluded'))
   if seed==42 and fold==1:
    with threadpool_limits(limits=1):fit=ppp.fit_points(coarse.loc[qt],events.loc[et],model.MODELS['M2_binary_contact'],.1)
    actual=pd.Series(ppp.log_intensity(fit,events.loc[test]),index=events.loc[test,'DEPOSIT'])
    err=max(abs(actual.loc[r.deposit]-r.log_intensity) for r in saved.itertuples())
    assert err<1e-7
    independent=coarse.loc[qt,'log1p_Cu_ppm'].to_numpy();weights=coarse.loc[qt,'area_km2'].to_numpy()
    independent=np.where(np.isfinite(independent),independent,fit.prep.medians['log1p_Cu_ppm'])
    mean=np.average(independent,weights=weights);assert abs(mean-fit.prep.means['log1p_Cu_ppm'])<1e-12
    replay=dict(seed=42,fold=1,max_saved_score_error=err,training_quadrature_count=int(qt.sum()),Cu_training_weighted_mean=mean,Cu_model_mean=fit.prep.means['log1p_Cu_ppm'])
 regions=gpd.read_file(RUN/'compact_regions.gpkg')
 ca=pd.read_csv(RUN/'compact_event_assignments.csv').set_index('DEPOSIT')
 labels=ca.loc[events.DEPOSIT,'fold'].to_numpy()
 cm=pd.read_csv(RUN/'compact_fold_metrics.csv');cm=cm[cm.model.eq('M2_binary_contact')]
 ch=pd.read_csv(RUN/'compact_held_out_events.csv');ch=ch[ch.model.eq('M2_binary_contact')]
 assert len(ch)==44 and ch.deposit.is_unique
 for row in cm.itertuples():
  reg=regions.loc[regions.fold.eq(row.fold),'geometry'].iloc[0]
  test=labels==row.fold
  train=(~test)&(shapely.distance(shapely.points(events.metric_x,events.metric_y),reg)>50000)
  sep=cdist(events.loc[train,['metric_x','metric_y']],events.loc[test,['metric_x','metric_y']]).min()/1000
  assert sep>50 and np.isclose(sep,row.minimum_event_separation_km)
  assert train.sum()==row.train_events and test.sum()==row.test_events
  assert set(events.loc[test,'DEPOSIT'])==set(ch[ch.fold.eq(row.fold)].deposit)
  folds.append(dict(method='compact',seed=42,fold=row.fold,train=int(train.sum()),test=int(test.sum()),guard_excluded=int((~train&~test).sum()),minimum_distance_km=sep,passed=True))
 save(pd.DataFrame(folds),'fold_checks.csv');save(pd.DataFrame(roles),'block_point_roles.csv')
 params=json.loads((RUN/'fitted_models.json').read_text(encoding='utf-8'))['M2_binary_contact']
 prep=ppp.legacy.Preprocessor(params['features'],params['medians'],params['means'],params['stds'])
 score=ppp.matrix(ext,prep)@np.array(params['slopes'])+params['intercept']
 bg=ppp.matrix(fine,prep)@np.array(params['slopes'])+params['intercept']
 threshold=ppp.weighted_quantile(bg,fine.area_km2.to_numpy(),.9)
 stored=pd.read_csv(RUN/'external_deposit_scores.csv');stored=stored[stored.model.eq('M2_binary_contact')].set_index('paper_star_id').loc[ext.paper_star_id]
 assert np.max(np.abs(score-stored.log_intensity.to_numpy()))<1e-10
 misses=ext.loc[score<threshold].copy();assert len(misses)==4
 misses['area_percentile']=stored.loc[misses.paper_star_id,'area_percentile'].to_numpy()
 misses['top_area_percent_needed']=100*(1-misses.area_percentile)
 misses['review_flag']=misses.paper_star_id.isin([22,26,27,39,40,44,48,56])
 cols=['paper_star_id','DEPOSIT','longitude','latitude','area_percentile','top_area_percent_needed','lith_intrusive','dist_binary_contact_km','nearest_geochemistry_km','has_gravity','review_flag']
 save(misses[cols],'four_missed_points.csv')
 details=[]
 for r in misses.itertuples():
  for f in params['features']:
   v=getattr(r,f);train=events[f].dropna()
   details.append(dict(deposit=r.DEPOSIT,feature=f,value=v,train_min=train.min(),train_max=train.max(),outside_training_event_range=bool(v<train.min() or v>train.max())))
 save(pd.DataFrame(details),'missed_point_feature_ranges.csv')
 checks={'coordinate_max_error_deg':float(delta.max()),'projection_max_error_m':float(projection_error.max()),'off_integration_centres_count':int((near>1e-6).sum()),'fold_checks':len(folds),'hash_checks':len(hashes),'replayed_fold':replay,'misses':misses[cols].to_dict('records')}
 (OUT/'audit_summary.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(checks,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
