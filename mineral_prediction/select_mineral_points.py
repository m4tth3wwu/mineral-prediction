"""Auditable occurrence selection scenarios and buffered cluster split manifests."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
import PPP_three_papers_simple as ppp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/point_selection_v1'
SOURCE=ROOT/'output/paper_point_matching/paper_figure2g_all_60_points_model_ready.csv'
FLAGS={22,26,27,39,40,44,48,56}

def clusters(xy,radius):
    """Connected distance groups are leakage controls, not geological identities."""
    return connected_components(cdist(xy,xy)<=radius,directed=False)[1]

def buffered_roles(xy,labels,group,guard):
    test=labels==group
    near=cdist(xy,xy[test]).min(axis=1)<=guard
    return np.where(test,'test',np.where(near,'guard_excluded','train'))

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    points=pd.read_csv(SOURCE)
    assert len(points)==60 and points.paper_star_id.is_unique
    points=points.sort_values('paper_star_id').reset_index(drop=True)
    points['review_flag']=points.paper_star_id.isin(FLAGS)
    points['identity_status']=np.where(points.review_flag,'requires_targeted_review','catalog_match_not_independently_confirmed')
    points['type_eligible']=points.CMMI_DEPOSIT_TYPE.fillna('').str.lower().str.startswith('porphyry copper')
    points['coordinate_valid']=np.isfinite(points.LONGITUDE)&np.isfinite(points.LATITUDE)&points.LONGITUDE.between(-180,180)&points.LATITUDE.between(-90,90)
    points['reason']=np.where(points.review_flag,'2026-09-05 documented identity/location questions','No current screening flag; NOT proof of correctness')
    points.to_csv(OUT/'selection_registry.csv',index=False,encoding='utf-8-sig')
    cache=ROOT/'output/ppp_binary_features_v3'
    events=pd.read_csv(cache/'events.csv').sort_values('paper_star_id').reset_index(drop=True)
    external=pd.read_csv(cache/'external.csv')
    grid=pd.read_csv(cache/'grid.csv')
    assert set(events.paper_star_id)==set(points.loc[points.paper_split=='training','paper_star_id'])
    assert set(external.paper_star_id)==set(points.loc[points.paper_split=='validation','paper_star_id'])
    all_features=pd.concat([events,external],ignore_index=True)
    xy_all=all_features[['metric_x','metric_y']].to_numpy()/1000
    distances=cdist(xy_all,xy_all)
    pairs=[]
    for i,j in zip(*np.where(np.triu(distances<=10,k=1))):
        pairs.append({'star_a':int(all_features.iloc[i].paper_star_id),
                      'star_b':int(all_features.iloc[j].paper_star_id),
                      'distance_km_projected':distances[i,j],
                      'action':'review only; proximity does not establish duplicate identity'})
    pd.DataFrame(pairs,columns=['star_a','star_b','distance_km_projected','action']).to_csv(OUT/'nearby_pairs_for_review.csv',index=False)
    # Use exactly the current M2 specification, avoiding an unsolicited model change.
    from PPP_binary_lithology_v3 import MODELS
    features=MODELS['M2_binary_contact']
    eligible=set(points.loc[points.coordinate_valid & points.type_eligible,'paper_star_id'])
    scenarios={'paper_baseline':set(events.paper_star_id),
               'catalog_eligible':set(events.paper_star_id)&eligible-{22},
               'review_flag_exclusion':(set(events.paper_star_id)&eligible)-FLAGS}
    summaries=[]; folds=[]; metrics=[]
    for name,ids in scenarios.items():
        train=events[events.paper_star_id.isin(ids)].copy()
        points[points.paper_star_id.isin(ids)].to_csv(OUT/f'{name}_training.csv',index=False,encoding='utf-8-sig')
        summaries.append({'scenario':name,'training_count':len(train),'fixed_external_count':len(external)})
        xy=train[['metric_x','metric_y']].to_numpy()/1000
        for radius in [10,25,50]:
            labels=clusters(xy,radius)
            for group in np.unique(labels):
                for guard in [25,50]:
                    roles=buffered_roles(xy,labels,group,guard)
                    if (roles=='train').any():
                        assert cdist(xy[roles=='train'],xy[roles=='test']).min()>guard
                    for sid,role in zip(train.paper_star_id,roles):
                        folds.append(dict(scenario=name,cluster_radius_km=radius,guard_km=guard,held_cluster=int(group),paper_star_id=int(sid),role=role,usable=bool((roles=='train').sum()>=10)))
        with threadpool_limits(limits=1):
            fit=ppp.fit_points(grid,train,features,alpha=0.1)
            for label,test in [('fixed_16',external),('same_unflagged_12',external[~external.paper_star_id.isin(FLAGS)])]:
                metrics.append({'scenario':name,'evaluation_set':label,'n_training':len(train),**ppp.evaluate(fit,grid,test)})
    pd.DataFrame(summaries).to_csv(OUT/'scenario_summary.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'buffered_cluster_roles.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'selection_sensitivity_metrics.csv',index=False)
    assert before==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    config=dict(version=1,target='porphyry copper including copper-molybdenum and copper-gold catalog types',
                flagged_ids=sorted(FLAGS),cluster_radii_km=[10,25,50],guards_km=[25,50],
                thresholds_status='predeclared engineering sensitivity choices, not literature universal constants',
                features=features,alpha=0.1,source_sha256=before,
                input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,cache/'events.csv',cache/'external.csv',cache/'grid.csv',ROOT/'select_mineral_points.py']},
                external_usage='historical comparison only; unchanged IDs, never used to choose occurrences',
                split_manifest_usage='point assignments only; a future PPP CV runner must also partition quadrature support and refit preprocessing within folds',
                identity_warning='No unflagged point is promoted to author-confirmed; proximity never automatically merges occurrences')
    (OUT/'selection_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summaries))

if __name__=='__main__': main()
