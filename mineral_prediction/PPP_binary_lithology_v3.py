"""Audited pink-vs-other lithology, paired PPP comparisons and spatial diagnostics.

Uses fixed 5 km quadrature support to isolate feature changes. This retains
the historical centre-mask area approximation, not exact coastline areas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, box
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

import PPP_three_papers_simple as ppp
import PPP_lithology_spatial_validation as legacy

ROOT = Path(__file__).resolve().parent
VERSION = 'pink_shared_contact_v3.1'
PINK = 'Igneous, intrusive'
EXCLUDED = {'Water', 'Unknown', 'Ice', 'Dam'}
NEW_DISTANCE = 'log1p_dist_binary_contact_km'
MODELS = {k: list(v) for k, v in ppp.MODELS.items() if not k.startswith('M3')}
MODELS['M2_old_polygon_contact'] = MODELS.pop('M2_intrusive_contact')
MODELS['M2_binary_contact'] = [NEW_DISTANCE if x == 'log1p_dist_intrusive_contact_km' else x
                              for x in MODELS['M2_old_polygon_contact']]
MODELS['M3_coverage_sensitivity'] = MODELS['M2_binary_contact'] + [ppp.PROXY]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def binary_geometry(lith):
    """Only shared mapped-rock class boundaries; no coast/no-data/crop edges.

Do not snap geometry: tolerances without source accuracy evidence can move
geological boundaries. Exact intersection can miss unmatched slivers; report it.
"""
    lith = lith.copy()
    lith.geometry = shapely.make_valid(lith.geometry.array)
    pink = shapely.union_all(lith.loc[lith.Symbol.eq(PINK)].geometry.array)
    other = shapely.union_all(lith.loc[~lith.Symbol.isin(EXCLUDED | {PINK})].geometry.array)
    overlap = pink.intersection(other).area
    # Match legacy priority at overlaps: pink takes precedence.
    other = other.difference(pink)
    shared = pink.boundary.intersection(other.boundary)
    lines = [g for g in shapely.get_parts(shared) if g.geom_type in ('LineString', 'MultiLineString')]
    shared = shapely.union_all(lines)
    if shared.is_empty or shared.length <= 0:
        raise ValueError('No shared pink/other rock contact; inspect source topology')
    old = shapely.union_all(lith.loc[lith.Symbol.eq(PINK)].geometry.boundary.array)
    stats = {'pink_area_km2': pink.area/1e6, 'other_area_km2': other.area/1e6,
             'class_overlap_before_priority_km2': overlap/1e6,
             'old_polygon_boundary_km': old.length/1000,
             'merged_pink_boundary_km': pink.boundary.length/1000,
             'shared_binary_contact_km': shared.length/1000,
             'excluded_merged_boundary_km': pink.boundary.difference(shared).length/1000,
             'snapping_tolerance_m': 0, 'excluded_contact_classes': sorted(EXCLUDED)}
    return pink, other, shared, old, stats


def nearest_lines(points, lines):
    geometries = [g for g in shapely.get_parts(lines) if g.length > 0]
    # Avoid a single huge geometry in every nearest-distance calculation.
    tree = shapely.STRtree(geometries)
    pairs, distance = tree.query_nearest(points, return_distance=True, all_matches=False)
    result = np.full(len(points), np.nan)
    result[pairs[0]] = distance / 1000
    if not np.isfinite(result).all():
        raise ValueError('Unresolved contact distances')
    return result


def feature_signature(cache_dir, cfg):
    files = [cache_dir/x for x in ['quadrature_grid_features_5km.csv',
             'training_deposit_features.csv', 'external_deposit_features.csv', 'config.json']]
    files += [cfg.geochemistry_path, cfg.gravity_path, Path(__file__),
              ROOT/'PPP_lithology_spatial_validation.py', ROOT/'PPP_three_papers_simple.py']
    for shp in [cfg.lithology_path, cfg.fault_path]:
        files += [shp.with_suffix(ext) for ext in ('.shp', '.shx', '.dbf', '.prj')]
    for name in ['paper_figure2g_training_44_model_ready.csv', 'paper_figure2g_validation_16_model_ready.csv']:
        files.append(cfg.paper_point_dir/name)
    logging.info('Hashing source data and code for cache validation')
    return {'version': VERSION, 'pink_symbol': PINK, 'excluded_contact_classes': sorted(EXCLUDED),
            'geometry_rule': 'make_valid_union_shared_boundary_pink_priority_no_snap',
            'base_features': 'recomputed_from_raw_on_fixed_legacy_support',
            'parameters': {'bounds':cfg.study_bounds, 'crs':cfg.equal_area_crs,
                           'idw_neighbors':cfg.idw_neighbors,'idw_power':cfg.idw_power,
                           'idw_floor_m':cfg.idw_floor_m,'gravity_max_distance_deg':cfg.gravity_max_distance_deg},
            'files': {str(p): digest(p) for p in files}}


def load_features(cache_dir, cache_out, cfg):
    signature = feature_signature(cache_dir, cfg)
    # Canonical JSON also normalizes tuples for exact cache comparisons.
    signature = json.loads(json.dumps(signature))
    manifest_path = cache_out/'manifest.json'
    names = ['grid.csv', 'events.csv', 'external.csv']
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        outputs = manifest.get('outputs', {})
        if manifest.get('signature') == signature and outputs and all(
                (cache_out/n).exists() and digest(cache_out/n) == h for n,h in outputs.items()):
            logging.info('Validated feature cache reused: %s', cache_out)
            return [pd.read_csv(cache_out/n, low_memory=False) for n in names]
        logging.info('Feature cache invalidated: source, code, rule or output changed')
    cache_out.mkdir(parents=True, exist_ok=True)
    fine, events, external, _ = ppp.load_features(cache_dir)
    # Check that the fixed support's event locations agree with the actual
    # current source tables; a changed event table must not silently use old coords.
    for cached, file in [(events,'paper_figure2g_training_44_model_ready.csv'),
                         (external,'paper_figure2g_validation_16_model_ready.csv')]:
        source = pd.read_csv(cfg.paper_point_dir/file).set_index('DEPOSIT').loc[cached.DEPOSIT]
        if not np.allclose(source[['LONGITUDE','LATITUDE']].to_numpy(float),
                           cached[['longitude','latitude']].to_numpy(float), atol=1e-9, rtol=0):
            raise ValueError('Event source changed: rebuild the legacy location/support cache first')
    lengths = [len(x) for x in (fine,events,external)]
    targets = pd.concat([fine,events,external], ignore_index=True)
    logging.info('Reading source rock polygons')
    lith = legacy.read_lithology(cfg)
    logging.info('Merging binary rock classes and extracting shared contacts')
    pink, other, contact, old, stats = binary_geometry(lith)
    points = shapely.points(targets.metric_x, targets.metric_y)
    targets['lith_intrusive'] = shapely.covers(pink, points).astype(np.int8)
    targets['lithology_valid'] = shapely.covers(pink.union(other), points)
    if not targets.iloc[len(fine):].lithology_valid.all():
        raise ValueError('Event outside valid binary rock coverage; review rather than code as other')
    targets['dist_intrusive_contact_km'] = nearest_lines(points, old)
    targets['dist_binary_contact_km'] = nearest_lines(points, contact)
    targets[NEW_DISTANCE] = np.log1p(targets.dist_binary_contact_km)
    logging.info('Recomputing geochemistry, fault and gravity from original sources')
    targets = legacy.add_geochemistry(targets, cfg)
    targets = legacy.add_fault_distance(targets, cfg)
    targets = legacy.add_gravity(targets, cfg)
    targets = legacy.add_transformed_features(targets)
    targets[ppp.PROXY] = np.log1p(targets.nearest_geochemistry_km.clip(lower=0))
    targets['contact_distance_change_km'] = targets.dist_binary_contact_km - targets.dist_intrusive_contact_km
    stats['valid_support_fraction'] = float(targets.iloc[:len(fine)].lithology_valid.mean())
    stats['grid_distance_change_median_km'] = float(targets.iloc[:len(fine)].contact_distance_change_km.median())
    stats['grid_distance_change_max_km'] = float(targets.iloc[:len(fine)].contact_distance_change_km.max())
    save_json(cache_out/'geometry_audit.json',stats)
    gpd.GeoDataFrame({'kind':['pink','other']},geometry=[pink,other],crs=cfg.equal_area_crs).to_file(cache_out/'binary_lithology.gpkg',driver='GPKG')
    gpd.GeoDataFrame({'kind':['shared_contact']},geometry=[contact],crs=cfg.equal_area_crs).to_file(cache_out/'binary_contacts.gpkg',driver='GPKG')
    frames = [x.copy().reset_index(drop=True) for x in np.split(targets,np.cumsum(lengths)[:-1])]
    for name,frame in zip(names,frames):
        csv(frame,cache_out/name)
    csv(targets.iloc[len(fine):][['DEPOSIT','lith_intrusive','lithology_valid','dist_intrusive_contact_km',
                                 'dist_binary_contact_km','contact_distance_change_km']],cache_out/'deposit_boundary_audit.csv')
    all_names = names + ['geometry_audit.json','binary_lithology.gpkg','binary_contacts.gpkg','deposit_boundary_audit.csv']
    save_json(manifest_path,{'signature':signature,'outputs':{n:digest(cache_out/n) for n in all_names}})
    return frames


def compact_regions(events, fine, folds=4):
    """Voronoi regions from k-means of training event coordinates, never scores.

No overclustering/balancing that recombines disjoint regions. Empty or tiny
training sets are reported instead of silently changing the geographic split.
"""
    centers = KMeans(n_clusters=folds, random_state=42, n_init=20).fit(
        events[['metric_x','metric_y']].to_numpy(float)).cluster_centers_
    xy = fine[['metric_x','metric_y']].to_numpy(float)
    bounds = box(xy[:,0].min()-1e6,xy[:,1].min()-1e6,xy[:,0].max()+1e6,xy[:,1].max()+1e6)
    parts = list(shapely.get_parts(shapely.voronoi_polygons(MultiPoint(centers),extend_to=bounds,ordered=True)))
    return centers,parts


def compact_validation(coarse, fine, events, out, alpha):
    centers,regions = compact_regions(events,fine)
    def labels(frame):
        xy=frame[['metric_x','metric_y']].to_numpy(float)
        return ((xy[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
    qfold,efold=labels(coarse),labels(events)
    qp=shapely.points(coarse.metric_x,coarse.metric_y)
    ep=shapely.points(events.metric_x,events.metric_y)
    rows,held=[],[]
    for fold,region in enumerate(regions):
        qtrain=(qfold!=fold)&(shapely.distance(qp,region)>50000)
        etrain=(efold!=fold)&(shapely.distance(ep,region)>50000)
        te=events.loc[efold==fold]
        if len(te)==0 or etrain.sum()<2:
            raise ValueError(f'Compact fold {fold+1} has insufficient events')
        distance=NearestNeighbors(n_neighbors=1).fit(te[['metric_x','metric_y']]).kneighbors(
            events.loc[etrain,['metric_x','metric_y']])[0].min()/1000
        if distance<=50:
            raise AssertionError('Compact fold buffer violation')
        for name,features in MODELS.items():
            fit=ppp.fit_points(coarse.loc[qtrain],events.loc[etrain],features,alpha)
            # Univariate novelty is a diagnostic, not a full joint-support test.
            xtrain=ppp.matrix(coarse.loc[qtrain],fit.prep)
            xtest=ppp.matrix(te,fit.prep)
            novel=((xtest<xtrain.min(axis=0))|(xtest>xtrain.max(axis=0))).any(axis=1)
            rows.append({'model':name,'seed':42,'fold':fold+1,'grid_km':10,
                         'train_events':int(etrain.sum()),'minimum_event_separation_km':float(distance),
                         'novel_test_event_fraction':float(novel.mean()),
                         **ppp.evaluate(fit,coarse.loc[qfold==fold],te)})
            for n,s in zip(te.DEPOSIT,ppp.log_intensity(fit,te)):
                held.append({'model':name,'fold':fold+1,'deposit':n,'log_intensity':s})
        logging.info('Compact region %s: %s training / %s test events',fold+1,etrain.sum(),len(te))
    csv(pd.DataFrame(rows),out/'compact_fold_metrics.csv')
    csv(ppp.summarize_cv(pd.DataFrame(rows))[1],out/'compact_cv_summary.csv')
    csv(pd.DataFrame(held),out/'compact_held_out_events.csv')
    csv(events[['DEPOSIT','longitude','latitude']].assign(fold=efold+1),out/'compact_event_assignments.csv')
    gpd.GeoDataFrame({'fold':np.arange(4)+1},geometry=regions,crs='ESRI:102039').to_file(out/'compact_regions.gpkg',driver='GPKG')
    return labels(fine),regions


def diagnostics(coarse,fine,events,fit,out,alpha):
    features=MODELS['M2_binary_contact']
    qkeys=ppp.block_keys(coarse,200)
    ekeys=ppp.block_keys(events,200)
    base_log=ppp.log_intensity(fit,fine)
    area=fine.area_km2.to_numpy(float)
    selected=base_log>=ppp.weighted_quantile(base_log,area,.9)
    influence=[]
    for i,row in events.iterrows():
        refit=ppp.fit_points(coarse,events.drop(index=i),features,alpha)
        scores=ppp.log_intensity(refit,fine)
        target=scores>=ppp.weighted_quantile(scores,area,.9)
        influence.append({'deposit':row.DEPOSIT,'paper_star_id':row.paper_star_id,
                          'top10_area_jaccard':float(area[target&selected].sum()/area[target|selected].sum()),
                          'slope_l2_change':float(np.linalg.norm(refit.slopes-fit.slopes))})
    csv(pd.DataFrame(influence).sort_values('top10_area_jaccard'),out/'event_influence.csv')
    count=np.zeros(len(fine),int)
    mean=np.zeros(len(fine));m2=np.zeros(len(fine));runs=[]
    for k,key in enumerate(sorted(set(ekeys)),1):
        # Entire block and 50 km guard removed from both events and integral.
        qt=ppp.outside_guard(coarse,[key],200,50)
        et=ppp.outside_guard(events,[key],200,50)
        refit=ppp.fit_points(coarse.loc[qt],events.loc[et],features,alpha)
        scores=ppp.log_intensity(refit,fine)
        # Normalize over the same area before comparing map shape.
        scores=scores-np.log(np.average(np.exp(scores),weights=area))
        target=scores>=ppp.weighted_quantile(scores,area,.9)
        count+=target
        delta=scores-mean;mean+=delta/k;m2+=delta*(scores-mean)
        runs.append({'block_x':key[0],'block_y':key[1],'train_events':int(et.sum()),
                     'top10_actual_area_fraction':float(area[target].sum()/area.sum())})
    csv(pd.DataFrame(runs),out/'stability_refits.csv')
    stable=fine[['cell_id','longitude','latitude','metric_x','metric_y','area_km2']].copy()
    stable['top10_selection_frequency']=count/len(runs)
    stable['normalized_log_intensity_sd']=np.sqrt(m2/max(1,len(runs)-1))
    csv(stable,out/'target_stability.csv')
    # These residuals concern counts in this selected catalogue, not undiscovered ore.
    residuals={key:{'block_x':key[0],'block_y':key[1],'observed':0,'predicted':0.,'area_km2':0.} for key in set(qkeys)|set(ekeys)}
    mass=coarse.area_km2.to_numpy(float)*np.exp(ppp.log_intensity(fit,coarse))
    for key,m,a in zip(qkeys,mass,coarse.area_km2):
        residuals[key]['predicted']+=m;residuals[key]['area_km2']+=a
    for key in ekeys:residuals[key]['observed']+=1
    residuals=pd.DataFrame(residuals.values())
    residuals['raw_residual']=residuals.observed-residuals.predicted
    csv(residuals,out/'regional_count_residuals.csv')
    return stable,residuals


def point_audit(out):
    directory=ROOT/'output/paper_point_matching'
    matches=pd.read_csv(directory/'paper_figure2g_star_matches.csv')
    matches['candidate_gap_km']=matches.second_distance_km-matches.match_distance_km
    matches['review_unmatched']=matches.match_status.ne('matched')
    matches['review_distance_over_10km']=matches.match_status.eq('matched') & matches.match_distance_km.gt(10)
    matches['review_candidate_gap_under_1km']=matches.match_status.eq('matched') & matches.candidate_gap_km.lt(1)
    matches['review_required']=matches[['review_unmatched','review_distance_over_10km','review_candidate_gap_under_1km']].any(axis=1)
    csv(matches,out/'paper_point_audit.csv')
    return matches


def plots(fine,events,external,folds,regions,cache,out,stable,residuals):
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    axes[0].scatter(fine.longitude,fine.latitude,c=np.where(fine.lith_intrusive.eq(1),'#ffbebe','#d8d8d8'),s=.4,rasterized=True)
    axes[0].scatter(events.longitude,events.latitude,s=15,c='black',label='44 training events')
    axes[0].scatter(external.longitude,external.latitude,s=18,c='#157f9c',marker='x',label='16 exploratory events')
    axes[0].set_title('Binary lithology: pink / other');axes[0].legend(fontsize=8)
    # Most changed training event is chosen from geometry only, not performance.
    r=events.iloc[int(events.contact_distance_change_km.argmax())]
    cfg=legacy.Config();lith=legacy.read_lithology(cfg)
    zoom=lith.cx[r.metric_x-40000:r.metric_x+40000,r.metric_y-40000:r.metric_y+40000]
    pink=zoom.loc[zoom.Symbol.eq(PINK)]
    if len(pink):
        pink.plot(ax=axes[1],color='#ffbebe',edgecolor='#999999',linewidth=.5)
    contact=gpd.read_file(cache/'binary_contacts.gpkg')
    contact.clip(box(r.metric_x-40000,r.metric_y-40000,r.metric_x+40000,r.metric_y+40000)).plot(ax=axes[1],color='#b1003a',linewidth=1.1)
    axes[1].scatter([r.metric_x],[r.metric_y],c='black',s=30)
    axes[1].set_xlim(r.metric_x-40000,r.metric_x+40000);axes[1].set_ylim(r.metric_y-40000,r.metric_y+40000)
    axes[1].set_title(f'Boundary audit near {r.DEPOSIT}\ngrey: original; red: binary contact',fontsize=10)
    fig.savefig(out/'binary_lithology_and_boundary.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    axes[0].scatter(fine.longitude,fine.latitude,c=folds,s=.4,cmap='tab10',rasterized=True)
    axes[0].scatter(events.longitude,events.latitude,c='black',s=12)
    axes[0].set_title('Four compact validation regions')
    im=axes[1].scatter(stable.longitude,stable.latitude,c=stable.top10_selection_frequency,s=.4,vmin=0,vmax=1,cmap='viridis',rasterized=True)
    fig.colorbar(im,ax=axes[1],label='Selection frequency (not probability)')
    axes[1].set_title('Top 10% target stability')
    lim=max(abs(residuals.raw_residual).max(),.1)
    im=axes[2].scatter((residuals.block_x+.5)*200,(residuals.block_y+.5)*200,c=residuals.raw_residual,s=35,marker='s',vmin=-lim,vmax=lim,cmap='RdBu_r')
    fig.colorbar(im,ax=axes[2],label='Observed - fitted catalogue events')
    axes[2].set_title('Regional count residuals');axes[2].set_aspect('equal')
    for ax in axes[:2]:ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
    fig.savefig(out/'validation_stability_diagnostics.png',dpi=180);plt.close(fig)
    # Every compact fold shows actual buffered training and held-out events.
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    points=shapely.points(events.metric_x,events.metric_y)
    for i,(ax,region) in enumerate(zip(axes.ravel(),regions)):
        inside=shapely.covers(region,points)
        train=shapely.distance(points,region)>50000
        ax.scatter(fine.metric_x[::20]/1000,fine.metric_y[::20]/1000,c=(folds[::20]==i),cmap='Pastel1',s=1)
        for mask,color,label in [(train,'#246d3a','train'),(inside,'#b5203d','test'),(~train&~inside,'#e4a52b','guard excluded')]:
            ax.scatter(events.loc[mask,'metric_x']/1000,events.loc[mask,'metric_y']/1000,c=color,s=20,label=f'{label}: {mask.sum()}')
        ax.set_title(f'Compact fold {i+1}, 50 km guard');ax.legend(fontsize=8);ax.set_aspect('equal')
    fig.savefig(out/'compact_fold_maps.png',dpi=160);plt.close(fig)


def report(out,audit):
    spatial=pd.read_csv(out/'spatial_cv_summary.csv')
    compact=pd.read_csv(out/'compact_cv_summary.csv')
    external=pd.read_csv(out/'external_validation_summary.csv')
    rows=['# 粉色二分类 PPP 改进版结果','',
          '岩性为粉色侵入岩=1、其余有效岩性=0；缺失/水体等不构造地质接触带。',
          '主对照在同一固定5 km面积支撑、10 km积分、200 km分块、50 km隔离、种子42/43/44、alpha=0.1下进行。',
          '四个连续验证区由44个训练矿点坐标的固定种子聚类及Voronoi分区生成；未按模型得分选择。','',
          '|模型|重复分块AUC|连续区域AUC|外部前10%命中|','|---|---:|---:|---:|']
    for _,row in spatial.iterrows():
        c=compact.loc[compact.model.eq(row.model)].iloc[0]
        e=external.loc[external.model.eq(row.model)].iloc[0]
        rows.append(f'|{row.model}|{row.auc_mean:.3f}|{c.auc_mean:.3f}|{int(e.top_10pct_hits)}/16|')
    rows += ['', '## 矿点来源审计','',
             f'60点中，按预先定义的匹配审计规则有 {audit.review_required.sum()} 点需重点复核（规则不是统计置信概率）。',
             '规则：未匹配、匹配距离>10 km、或第一/第二候选距离差<1 km。距离差阈值只是排查规则，不是地理定位误差估计。',
             '原44/16划分和60个记录均保留；没有凭低预测分数替换矿点。详见 paper_point_audit.csv。','',
             '## 诊断解释与限制','',
             '- event_influence.csv：逐个删除训练矿点重拟合，报告前10%靶区面积交并比和系数变化。',
             '- target_stability.csv：逐个留出含矿点的200 km块及50 km隔离带后，重拟合并在全区预测的前10%入选频率；不是置信区间或有矿概率。',
             '- regional_count_residuals.csv：所选目录事件数的拟合残差，不是未知矿床数量误差。',
             '- 连续区测试特征超出训练区逐变量范围的比例列于 compact_fold_metrics.csv；不等于完整多变量适用域分析。',
             '- 16个外部点在历史上已用于多轮比较，本轮仅探索性评价。',
             '- 网格海岸面积仍为中心筛选近似；保留Kelsey附近已知小格缺口，并单独记录。',
             '- 几何修复不作任意距离吸附；共享边界可能遗漏不吻合图斑缝隙，几何审计记录边长和重叠面积。',
             '- 地化、断层、重力在固定目标位置从原始数据重算；缓存校验原始文件、代码、参数与输出指纹。',
             '- 区域协变量作为事先可用的调查数据，不能声称预测完全没有调查的新地区；真实勘查选择偏差仍未校正。','',
             '## 工程参考','',
             '- https://geopandas.org/en/latest/docs/reference/api/geopandas.GeoDataFrame.dissolve.html',
             '- https://github.com/rvalavi/blockCV',
             '- https://github.com/spatstat/spatstat.model']
    (out/'README_results.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/ppp_binary_lithology_v3')
    parser.add_argument('--feature-cache',type=Path,default=ROOT/'output/ppp_binary_features_v3')
    parser.add_argument('--legacy-cache',type=Path,default=ROOT/'output/ppp_lithology_spatial_validation_5km')
    args=parser.parse_args();out=args.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh output directory')
    legacy.setup_logging(out)
    with threadpool_limits(limits=1):
        fine,events,external=load_features(args.legacy_cache,args.feature_cache,legacy.Config())
        save_json(out/'config.json',{'version':VERSION,'alpha':.1,'seeds':[42,43,44],
                                   'feature_manifest_sha256':digest(args.feature_cache/'manifest.json'),
                                   'feature_cache':str(args.feature_cache),'models':MODELS})
        csv(ppp.boundary_audit(fine,events,external),out/'boundary_quadrature_audit.csv')
        coarse=ppp.coarsen_quadrature(fine,10)
        old_models=ppp.MODELS
        try:
            ppp.MODELS=MODELS
            rows,assignments,held=ppp.spatial_validation(coarse,fine,events,10,200,50,5,[42,43,44],.1)
        finally:ppp.MODELS=old_models
        repeats,summary=ppp.summarize_cv(rows)
        for name,frame in [('spatial_fold_metrics',rows),('spatial_block_assignments',assignments),
                           ('held_out_event_predictions',held),('spatial_repeat_metrics',repeats),('spatial_cv_summary',summary)]:
            csv(frame,out/f'{name}.csv')
        labels,regions=compact_validation(coarse,fine,events,out,.1)
        fits={name:ppp.fit_points(coarse,events,features,.1) for name,features in MODELS.items()}
        pred=fine[['cell_id','longitude','latitude','metric_x','metric_y','area_km2','lith_intrusive','lithology_valid']].copy()
        parameters={}
        for name,fit in fits.items():
            pred[name+'_intensity']=np.exp(ppp.log_intensity(fit,fine))
            parameters[name]={'intercept':fit.intercept,'slopes':fit.slopes.tolist(),
                              'features':fit.prep.features,'medians':fit.prep.medians,'means':fit.prep.means,'stds':fit.prep.stds}
        csv(pred,out/'predictions_5km.csv');save_json(out/'fitted_models.json',parameters)
        csv(pd.concat([ppp.coefficients(n,f) for n,f in fits.items()]),out/'coefficients.csv')
        logging.info('Computing single-event influence and block-deletion target stability')
        stable,residuals=diagnostics(coarse,fine,events,fits['M2_binary_contact'],out,.1)
        audit=point_audit(out)
        # External evaluation only after fixed training-only analyses.
        ext=[];scores=[]
        for name,fit in fits.items():
            ext.append({'model':name,**ppp.evaluate(fit,fine,external)})
            lq=ppp.log_intensity(fit,fine);le=ppp.log_intensity(fit,external)
            scores.append(external[['DEPOSIT','paper_star_id','longitude','latitude']].assign(model=name,log_intensity=le,
                area_percentile=[np.average(lq<=s,weights=fine.area_km2) for s in le]))
        csv(pd.DataFrame(ext),out/'external_validation_summary.csv');csv(pd.concat(scores),out/'external_deposit_scores.csv')
        logging.info('Rendering result maps')
        plots(fine,events,external,labels,regions,args.feature_cache,out,stable,residuals)
        report(out,audit)
        logging.info('Completed %s',out)


if __name__=='__main__':main()
