"""Small, reproducible presence-only experiment; leaves the old ACAWLR/PPP intact.

Phillips et al. 2009: audit sampling coverage, do not invent true absences.
Renner et al. 2015: exact event coordinates + area-weighted numerical integral.
Roberts et al. 2017: repeated spatial blocks with a geometric exclusion buffer.

Input: the existing 5 km feature cache. Its land mask and geological features
remain approximations. This is NOT a full survey-bias correction or a Cox model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

import PPP_lithology_spatial_validation as legacy

ROOT = Path(__file__).resolve().parent
PROXY = "log1p_nearest_geochemistry_km"
MODELS = {name: list(features) for name, features in legacy.MODEL_VARIANTS.items()}
MODELS["M3_sampling_coverage_sensitivity"] = [*MODELS["M2_intrusive_contact"], PROXY]


def weighted_quantile(values, weights, fraction):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if not 0 <= fraction <= 1 or len(values) == 0:
        raise ValueError("Invalid quantile or empty sample")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Quantiles require finite values and positive weights")
    order = np.argsort(values, kind="stable")
    pos = np.searchsorted(np.cumsum(weights[order]), fraction * weights.sum(), side="left")
    return float(values[order[min(pos, len(order) - 1)]])


def boundary_audit(grid, events, external):
    keys = set(zip(*np.floor(grid[["metric_x", "metric_y"]].to_numpy(float).T / 5000).astype(int)))
    rows = []
    nearest = NearestNeighbors(n_neighbors=1).fit(grid[["metric_x", "metric_y"]])
    for group, frame in (("training_44", events), ("external_16", external)):
        point_keys = list(zip(*np.floor(frame[["metric_x", "metric_y"]].to_numpy(float).T / 5000).astype(int)))
        outside = [i for i, key in enumerate(point_keys) if key not in keys]
        if outside:
            distance = nearest.kneighbors(frame.iloc[outside][["metric_x", "metric_y"]])[0].ravel() / 1000
            for index, dist in zip(outside, distance):
                rows.append({"group": group, "deposit": frame.iloc[index].DEPOSIT,
                             "lith_symbol": frame.iloc[index].lith_symbol,
                             "nearest_quadrature_km": float(dist)})
    return pd.DataFrame(rows, columns=["group", "deposit", "lith_symbol", "nearest_quadrature_km"])


def validate_inputs(grid, events, external):
    required = set(sum(MODELS.values(), [])) | {"metric_x", "metric_y", "nearest_geochemistry_km"}
    for name, frame in (("grid", grid), ("events", events), ("external", external)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name}: missing {required - set(frame.columns)}")
        if not np.isfinite(frame[["metric_x", "metric_y"]].to_numpy(float)).all():
            raise ValueError(f"{name}: nonfinite coordinates")
        if frame[["metric_x", "metric_y"]].duplicated().any():
            raise ValueError(f"{name}: duplicate coordinates need manual review")
    if len(events) != 44 or len(external) != 16:
        raise ValueError("This experiment expects the existing 44/16 deposits")
    if not np.isfinite(grid.area_km2).all() or (grid.area_km2 <= 0).any():
        raise ValueError("Quadrature areas must be finite and positive")
    combined = pd.concat([events, external], ignore_index=True)
    if combined[["metric_x", "metric_y"]].duplicated().any():
        raise ValueError("Train/external coordinate overlap")
    names = combined.DEPOSIT.astype(str).str.strip().str.casefold()
    if names.duplicated().any():
        raise ValueError("Repeated deposit names need manual review")
    if combined.lith_symbol.isin(["Water", "Unknown"]).any():
        raise ValueError("A deposit lies outside the mapped land domain")
    # The actual study region is mapped land, not the union of squares whose
    # centres passed the old land filter. Audit boundary quadrature gaps, never
    # snap or silently delete points. Known land points near a gap are allowed.
    boundary = boundary_audit(grid, events, external)
    if not boundary.empty:
        if (boundary.nearest_quadrature_km > 5 * np.sqrt(2)).any():
            raise ValueError(f"Land events too far from the quadrature support: {boundary.to_dict('records')}")
        logging.warning("Mapped-land events in a centre-mask boundary gap; retaining exact coordinates: %s",
                        boundary.to_dict("records"))


def load_features(cache_dir):
    paths = [cache_dir / "quadrature_grid_features_5km.csv",
             cache_dir / "training_deposit_features.csv", cache_dir / "external_deposit_features.csv"]
    config_path = cache_dir / "config.json"
    if not all(p.exists() for p in [*paths, config_path]):
        raise FileNotFoundError("Run the existing 5 km PPP feature pipeline first, or specify --cache-dir")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["grid_km"] != 5 or cfg["equal_area_crs"] != "ESRI:102039":
        raise ValueError("Expected a 5 km ESRI:102039 cache")
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    for frame in frames:
        frame[PROXY] = np.log1p(frame.nearest_geochemistry_km.clip(lower=0))
    validate_inputs(*frames)
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths, config_path]}
    return (*frames, manifest)


def coarsen_quadrature(fine, spacing_km):
    """Aggregate areas, choose an actual fine-grid point per larger cell.

    Both resolutions use the SAME approximate land-area quadrature. Coast area
    remains approximate at 5 km; square cells do not define the exact domain.
    """
    if spacing_km < 5 or spacing_km % 5 != 0:
        raise ValueError("Quadrature spacing must be a multiple of 5 km")
    out = fine.copy().reset_index(drop=True)
    xy = out[["metric_x", "metric_y"]].to_numpy(float)
    keys = np.floor(xy / (spacing_km * 1000)).astype(int)
    out["_gx"], out["_gy"] = keys[:, 0], keys[:, 1]
    centre = (keys + 0.5) * spacing_km * 1000
    out["_distance"] = ((xy - centre) ** 2).sum(axis=1)
    areas = out.groupby(["_gx", "_gy"]).area_km2.sum()
    out = out.sort_values(["_gx", "_gy", "_distance", "cell_id"], kind="stable")
    out = out.drop_duplicates(["_gx", "_gy"]).set_index(["_gx", "_gy"])
    out["area_km2"] = areas.reindex(out.index)
    out = out.reset_index(drop=True).drop(columns="_distance")
    # Counts are not observations for the new model. Prevent accidental reuse.
    out = out.drop(columns="event_count", errors="ignore")
    out["cell_id"] = np.arange(len(out))
    assert np.isclose(out.area_km2.sum(), fine.area_km2.sum())
    return out


def preprocess(grid, features):
    """Only training quadrature supplies area-weighted imputation/scaling."""
    weights = grid.area_km2.to_numpy(float)
    medians, means, stds = {}, {}, {}
    for feature in features:
        values = pd.to_numeric(grid[feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        median = weighted_quantile(values[valid], weights[valid], 0.5) if valid.any() else 0.0
        values = np.where(valid, values, median)
        mean = float(np.average(values, weights=weights))
        std = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
        if feature in legacy.BINARY_FEATURES:
            mean, std = 0.0, 1.0
        medians[feature], means[feature], stds[feature] = median, mean, max(std, 1e-12)
        if std < 1e-12:
            stds[feature] = 1.0
    return legacy.Preprocessor(features, medians, means, stds)


def matrix(frame, prep):
    cols = []
    for feature in prep.features:
        v = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        v = np.where(np.isfinite(v), v, prep.medians[feature])
        cols.append((v - prep.means[feature]) / prep.stds[feature])
    return np.column_stack(cols)


def profiled_objective(slopes, xq, event_sum, log_area, n_events, alpha):
    """Negative PPP log likelihood with the unpenalized intercept profiled out."""
    log_z = logsumexp(log_area + xq @ slopes)
    probabilities = np.exp(log_area + xq @ slopes - log_z)
    # Additive constants independent of slopes are omitted here.
    value = n_events * log_z - event_sum @ slopes + 0.5 * alpha * (slopes @ slopes)
    gradient = n_events * (xq.T @ probabilities) - event_sum + alpha * slopes
    return float(value), gradient


@dataclass
class PointFit:
    slopes: np.ndarray
    intercept: float
    prep: legacy.Preprocessor
    log_likelihood: float
    n_events: int
    converged: bool
    gradient_max: float


def fit_points(grid, events, features, alpha=0.1):
    if len(events) < 2 or grid.empty or alpha < 0:
        raise ValueError("Need at least two events, nonempty integration region and alpha >= 0")
    prep = preprocess(grid, features)
    xq, xe = matrix(grid, prep), matrix(events, prep)
    log_area = np.log(grid.area_km2.to_numpy(float))
    result = minimize(profiled_objective, np.zeros(len(features)),
                      args=(xq, xe.sum(axis=0), log_area, len(events), alpha),
                      method="L-BFGS-B", jac=True,
                      options={"maxiter": 1500, "ftol": 1e-13, "gtol": 1e-7, "maxls": 40})
    grad = float(np.max(np.abs(result.jac)))
    if not result.success or grad > 0.01:
        raise RuntimeError(f"PPP optimizer did not converge adequately: {result.message}; gradient={grad}")
    intercept = float(np.log(len(events)) - logsumexp(log_area + xq @ result.x))
    ll = float(len(events) * intercept + xe.sum(axis=0) @ result.x - len(events))
    return PointFit(result.x, intercept, prep, ll, len(events), bool(result.success), grad)


def log_intensity(fit, frame, standardize_coverage=False):
    x = matrix(frame, fit.prep)
    if standardize_coverage and PROXY in fit.prep.features:
        # Hold the nuisance proxy to the TRAINING DOMAIN's weighted mean in
        # log1p-distance units. This map is relative, not a calibrated true rate.
        x[:, fit.prep.features.index(PROXY)] = 0.0
    return fit.intercept + x @ fit.slopes


def evaluate(fit, grid, events, standardize_coverage=False):
    lq = log_intensity(fit, grid, standardize_coverage)
    le = log_intensity(fit, events, standardize_coverage)
    area = grid.area_km2.to_numpy(float)
    n = len(events)
    mass = float(np.exp(logsumexp(np.log(area) + lq)))
    auc = roc_auc_score(np.r_[np.ones(n), np.zeros(len(grid))], np.r_[le, lq],
                        sample_weight=np.r_[np.ones(n), area])
    result = {"presence_background_auc": float(auc), "test_events": n,
              "test_area_km2": float(area.sum()), "predicted_events": mass,
              "point_log_likelihood": float(le.sum() - mass),
              # Conditional-on-N spatial density score relative to a uniform
              # density on this same test region; it measures ranking/location,
              # not calibration of absolute unknown mineral resources.
              "conditional_log_gain_per_event": float(le.mean() - np.log(mass) + np.log(area.sum()))}
    for fraction in (0.05, 0.10):
        threshold = weighted_quantile(lq, area, 1 - fraction)
        label = int(100 * fraction)
        result[f"top_{label}pct_hits"] = int(np.sum(le >= threshold))
        result[f"top_{label}pct_actual_area_fraction"] = float(area[lq >= threshold].sum() / area.sum())
    return result


def block_keys(frame, block_km):
    return [tuple(v) for v in np.floor(frame[["metric_x", "metric_y"]].to_numpy(float) /
                                      (block_km * 1000)).astype(int)]


def make_fold_map(fine, events, block_km, folds, seed):
    """Stable across quadrature resolutions: always use the fixed fine mask."""
    area = {}
    for key, weight in zip(block_keys(fine, block_km), fine.area_km2):
        area[key] = area.get(key, 0.0) + weight
    counts = {key: 0 for key in area}
    for key in block_keys(events, block_km):
        if key not in area:
            raise ValueError("Event block is outside the integration domain")
        counts[key] += 1
    positive = [k for k in sorted(area) if counts[k] > 0]
    if len(positive) < folds:
        raise ValueError("Insufficient event-bearing blocks for spatial validation")
    rng = np.random.default_rng(seed)
    ties = {key: rng.random() for key in sorted(area)}
    positive.sort(key=lambda key: (-counts[key], ties[key]))
    fold_events, fold_area = np.zeros(folds), np.zeros(folds)
    mapping = {}
    for key in positive:
        target = min(range(folds), key=lambda f: (fold_events[f], fold_area[f]))
        mapping[key] = target
        fold_events[target] += counts[key]
        fold_area[target] += area[key]
    remaining = [k for k in sorted(area) if counts[k] == 0]
    remaining.sort(key=lambda key: (-area[key], ties[key]))
    for key in remaining:
        target = int(np.argmin(fold_area))
        mapping[key] = target
        fold_area[target] += area[key]
    return mapping


def fold_ids(frame, mapping, block_km):
    return np.array([mapping[k] for k in block_keys(frame, block_km)])


def outside_guard(frame, test_blocks, block_km, buffer_km):
    """Euclidean distance from true positions to held-out square BLOCKS.

    Geometry is exact for blocks, unlike distance to sampled test-grid centres.
    Numerical integration still selects quadrature cells by their point centres.
    """
    xy = frame[["metric_x", "metric_y"]].to_numpy(float)
    side = block_km * 1000
    keep = np.ones(len(frame), bool)
    for bx, by in test_blocks:
        low = np.array([bx, by]) * side
        high = low + side
        delta = np.maximum(np.maximum(low - xy, xy - high), 0.0)
        keep &= (delta ** 2).sum(axis=1) > (buffer_km * 1000) ** 2
    return keep


def spatial_validation(grid, fine, events, spacing, block_km, buffer_km, folds, seeds, alpha):
    rows, assignments, point_predictions = [], [], []
    for seed in seeds:
        mapping = make_fold_map(fine, events, block_km, folds, seed)
        qfold, efold = fold_ids(grid, mapping, block_km), fold_ids(events, mapping, block_km)
        for key, fold in sorted(mapping.items()):
            assignments.append({"seed": seed, "block_x": key[0], "block_y": key[1], "fold": fold + 1})
        for fold in range(folds):
            test_blocks = [key for key, assigned in mapping.items() if assigned == fold]
            qtrain = (qfold != fold) & outside_guard(grid, test_blocks, block_km, buffer_km)
            etrain = (efold != fold) & outside_guard(events, test_blocks, block_km, buffer_km)
            tq, te = grid.loc[qfold == fold], events.loc[efold == fold]
            if te.empty or len(events.loc[etrain]) < 2:
                raise ValueError("Empty test fold or insufficient buffered training events")
            min_distance = float(NearestNeighbors(n_neighbors=1).fit(te[["metric_x", "metric_y"]])
                                 .kneighbors(events.loc[etrain, ["metric_x", "metric_y"]])[0].min() / 1000)
            if min_distance <= buffer_km:
                raise AssertionError("Training/test event separation violated")
            for name, features in MODELS.items():
                fit = fit_points(grid.loc[qtrain], events.loc[etrain], features, alpha)
                row = {"model": name, "seed": seed, "fold": fold + 1, "grid_km": spacing,
                       "block_km": block_km, "buffer_km": buffer_km,
                       "train_events": int(etrain.sum()), "train_quadrature": int(qtrain.sum()),
                       "minimum_event_separation_km": min_distance, "optimizer_gradient_max": fit.gradient_max,
                       **evaluate(fit, tq, te)}
                rows.append(row)
                for deposit, score in zip(te.DEPOSIT, log_intensity(fit, te)):
                    point_predictions.append({"model": name, "seed": seed, "fold": fold + 1,
                                              "deposit": deposit, "log_intensity": float(score)})
            logging.info("%g km | seed %s fold %s | train %s test %s events | separation %.1f km",
                         spacing, seed, fold + 1, etrain.sum(), len(te), min_distance)
    return pd.DataFrame(rows), pd.DataFrame(assignments), pd.DataFrame(point_predictions)


def summarize_cv(rows):
    repeats = []
    for (model, seed, spacing), group in rows.groupby(["model", "seed", "grid_km"]):
        weights = group.test_events * group.test_area_km2
        repeats.append({"model": model, "seed": seed, "grid_km": spacing,
                        "same_fold_auc": np.average(group.presence_background_auc, weights=weights),
                        "top_10pct_capture": group.top_10pct_hits.sum() / group.test_events.sum(),
                        "conditional_log_gain_per_event": np.average(group.conditional_log_gain_per_event,
                                                                      weights=group.test_events),
                        "test_events": int(group.test_events.sum())})
    repeats = pd.DataFrame(repeats)
    summary = repeats.groupby(["model", "grid_km"]).agg(
        repeats=("seed", "size"), auc_mean=("same_fold_auc", "mean"),
        auc_min=("same_fold_auc", "min"), auc_max=("same_fold_auc", "max"),
        top_10pct_capture_mean=("top_10pct_capture", "mean"),
        log_gain_mean=("conditional_log_gain_per_event", "mean")).reset_index()
    return repeats, summary


def coverage_audit(grid, events, external):
    rows = []
    for name, frame in (("domain_area", grid), ("training_44", events), ("external_16", external)):
        weights = frame.area_km2.to_numpy(float) if name == "domain_area" else np.ones(len(frame))
        for feature in ("nearest_geochemistry_km", "gravity_distance_deg"):
            for fraction in (0.1, 0.5, 0.9):
                rows.append({"group": name, "feature": feature, "quantile": fraction,
                             "value": weighted_quantile(frame[feature], weights, fraction)})
    return pd.DataFrame(rows)


def coefficients(name, fit):
    rows = [{"model": name, "term": "intercept", "standardized_coefficient": fit.intercept,
             "raw_coefficient": fit.intercept - sum(fit.slopes[i] * fit.prep.means[f] / fit.prep.stds[f]
                                                    for i, f in enumerate(fit.prep.features))}]
    for i, feature in enumerate(fit.prep.features):
        rows.append({"model": name, "term": feature, "standardized_coefficient": fit.slopes[i],
                     "raw_coefficient": fit.slopes[i] / fit.prep.stds[feature]})
    return pd.DataFrame(rows)


def write_report(out, summary, external_rows, convergence, audit, boundary, args):
    lines = ["# 前三篇论文：本科科研简化版试验", "",
             "## 实际更改", "",
             "1. 直接使用 44 个矿点的真实坐标和当地特征计算点过程似然；不再归并成网格计数。",
             "2. 保留原来的地化、断层、重力和岩性对照；增加 M3：M2 + 距最近地化采样点的距离。",
             "   此距离仅为采样覆盖的敏感性代理，不是真实勘查强度，也没有生成确定无矿的负样本。",
             "3. 使用固定 5 km 陆地积分网格，聚合面积生成 10 km 积分点；两种积分分辨率近似总面积完全一致。",
             "4. 按真实坐标分配空间块，直接计算到测试方块的隔离距离，重复五折验证。",
             "5. 只在训练区估计缺失值填补、标准化；固定正则化，不用测试集挑参数。",
             "6. 增加留出矿点的点过程似然、相对均匀分布的条件对数得分和面积加权 AUC。", "",
             f"设置：{args.block_km:g} km 空间块，{args.buffer_km:g} km 隔离，种子 {args.seeds}；"
             f"alpha={args.alpha:g}。这些是预先固定的简化设置，不是论文推荐的通用距离。", "",
             "## 10 km 重复空间验证", "",
             "AUC 只比较同一折模型的分数，再按矿点数 × 测试区面积加权；"
             "范围是重复划分的最小值–最大值，不是置信区间。背景是区域面积，不是核实无矿地点。", "",
             "|模型|平均 AUC|重复范围|前 10% 面积平均捕获率|条件对数增益/矿点|",
             "|---|---:|---:|---:|---:|"]
    for r in summary.itertuples(index=False):
        lines.append(f"|{r.model}|{r.auc_mean:.3f}|{r.auc_min:.3f}–{r.auc_max:.3f}|"
                     f"{r.top_10pct_capture_mean:.1%}|{r.log_gain_mean:.3f}|")
    lines += ["", "## 16 个外部矿点：探索性比较", "",
              "16 点未参与拟合、标准化或空间分折，但在此前研究中已经用于方案比较，不能视为全新盲测。", "",
              "|模型|存在–背景 AUC|前 5% 面积命中|前 10% 面积命中|", "|---|---:|---:|---:|"]
    for r in external_rows.itertuples(index=False):
        lines.append(f"|{r.model}|{r.presence_background_auc:.3f}|{r.top_5pct_hits}/16|{r.top_10pct_hits}/16|")
    lines += ["", "## 积分精度检查", "",
              "固定矿点、研究区和特征定义，只改变数值积分点密度；两种拟合都在同一 5 km 网格上评估。",
              "|模型|10 km 拟合总量在细网格上的相对误差|两种拟合排序相关|前10%区域面积交并比|",
              "|---|---:|---:|---:|"]
    for r in convergence.itertuples(index=False):
        lines.append(f"|{r.model}|{r.coarse_integral_relative_error:.2%}|{r.map_spearman:.3f}|{r.top10_area_jaccard:.3f}|")
    lines += ["", "## 采样覆盖审计", "", "以下距离单位为 km；全区统计按面积加权。", "",
              "|对象|到最近地化采样点的中位距离|", "|---|---:|"]
    part = audit.loc[(audit.feature == "nearest_geochemistry_km") & (audit["quantile"] == 0.5)]
    for r in part.itertuples(index=False):
        lines.append(f"|{r.group}|{r.value:.2f}|")
    lines += ["", "## 没有做、不能声称的事", "",
              "- 未取得真实勘查努力或共享调查过程的目标群数据，不能声称已消除发现/收录偏差。",
              "- M3 的采样覆盖标准化图把该代理固定在训练区的平均 log1p 距离，只是敏感性图，不能当无偏真实矿床强度。",
              "- 44 点来自既有论文匹配子集，尚未证明它们代表完整清查；结果不等于未知矿床总数或有矿概率。",
              "- 陆地、海岸和隔离边界的面积仍由网格近似；没有执行真实地质多边形的精确面积裁切。",
              f"- {len(boundary)} 个陆地矿点位于旧网格中心筛选遗漏的小格，保留真实位置；详见 boundary_quadrature_audit.csv。",
              "- 地化插值图等视为预先可用的调查资料；验证不代表转移到完全没有这些资料的地区。",
              "- 岩性边界仍是原版侵入岩图斑边界，可能含内部制图边界；本次没有重建真实地质接触带。",
              "- 没有增加复杂神经网络、空间随机场、自动特征筛选或大规模调参。",
              "- 新旧流程的事件位置和评价定义不同，不把旧版 AUC 与新版 AUC 直接当作公平优劣比较。", "",
              "## 方法依据", "",
              "- [Phillips et al. 2009：采样偏差](https://doi.org/10.1890/07-2153.1)",
              "- [Renner et al. 2015：点过程与积分](https://doi.org/10.1111/2041-210X.12352)",
              "- [Roberts et al. 2017：结构化验证](https://doi.org/10.1111/ecog.02881)"]
    (out / "README_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "output/ppp_lithology_spatial_validation_5km")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/ppp_three_papers_simple_v2")
    parser.add_argument("--block-km", type=float, default=200)
    parser.add_argument("--buffer-km", type=float, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--fine-cv", action="store_true", help="Also evaluate 5 km quadrature with the first spatial split")
    args = parser.parse_args()
    args.seeds = [int(v) for v in args.seeds.split(",")]
    if (len(set(args.seeds)) != len(args.seeds) or not args.seeds or args.folds < 2
            or not np.isfinite([args.block_km, args.buffer_km, args.alpha]).all()
            or args.block_km <= 0 or args.block_km % 10 != 0 or args.buffer_km < 0 or args.alpha < 0):
        parser.error("Use unique seeds, >=2 folds, a block size divisible by 10 km, and nonnegative buffer/alpha")
    return args


def main():
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {out}; specify a fresh --output-dir")
    legacy.setup_logging(out)
    fine, events, external, manifest = load_features(args.cache_dir)
    boundary = boundary_audit(fine, events, external)
    boundary.to_csv(out / "boundary_quadrature_audit.csv", index=False, encoding="utf-8-sig")
    for source in (Path(__file__), ROOT / "PPP_lithology_spatial_validation.py"):
        manifest[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    fine = coarsen_quadrature(fine, 5)
    coarse = coarsen_quadrature(fine, 10)
    logging.info("Exact events %s; fine/coarse points %s/%s; identical area %.0f km2",
                 len(events), len(fine), len(coarse), fine.area_km2.sum())
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(input_sha256=manifest, algorithm="exact_events_ppp_v1",
                    fine_points=len(fine), coarse_points=len(coarse), area_km2=float(fine.area_km2.sum()),
                    papers=["10.1890/07-2153.1", "10.1111/2041-210X.12352", "10.1111/ecog.02881"])
    (out / "config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    with threadpool_limits(limits=1):
        rows, assignments, held_out = spatial_validation(coarse, fine, events, 10, args.block_km,
                                                        args.buffer_km, args.folds, args.seeds, args.alpha)
        repeats, summary = summarize_cv(rows)
        for filename, frame in (("spatial_fold_metrics", rows), ("spatial_repeat_metrics", repeats),
                                ("spatial_cv_summary", summary), ("spatial_block_assignments", assignments),
                                ("held_out_event_predictions", held_out)):
            frame.to_csv(out / f"{filename}.csv", index=False, encoding="utf-8-sig")
        if args.fine_cv:
            fine_rows, _, _ = spatial_validation(fine, fine, events, 5, args.block_km, args.buffer_km,
                                                args.folds, args.seeds[:1], args.alpha)
            fine_rows.to_csv(out / "fine_spatial_fold_metrics.csv", index=False, encoding="utf-8-sig")
            summarize_cv(fine_rows)[0].to_csv(out / "fine_spatial_repeat_metrics.csv", index=False, encoding="utf-8-sig")
        fitted, convergence, coeffs, fit_parameters = {}, [], [], {}
        predictions = fine[["cell_id", "longitude", "latitude", "metric_x", "metric_y", "area_km2",
                            "lith_intrusive", "nearest_geochemistry_km"]].copy()
        for name, features in MODELS.items():
            fit = fit_points(coarse, events, features, args.alpha)
            finer = fit_points(fine, events, features, args.alpha)
            fitted[name] = fit
            fit_parameters[name] = {"intercept": fit.intercept, "slopes": fit.slopes.tolist(),
                                    "features": fit.prep.features, "medians": fit.prep.medians,
                                    "means": fit.prep.means, "stds": fit.prep.stds,
                                    "training_events": fit.n_events, "training_log_likelihood": fit.log_likelihood}
            coeffs.append(coefficients(name, fit))
            lcoarse, lfine = log_intensity(fit, fine), log_intensity(finer, fine)
            area = fine.area_km2.to_numpy(float)
            selc = lcoarse >= weighted_quantile(lcoarse, area, 0.9)
            selfine = lfine >= weighted_quantile(lfine, area, 0.9)
            coarse_mass = float(np.exp(logsumexp(np.log(area) + lcoarse)))
            convergence.append({"model": name, "coarse_points": len(coarse), "fine_points": len(fine),
                                "area_km2": float(area.sum()),
                                "coarse_integral_relative_error": coarse_mass / len(events) - 1,
                                "coarse_ll_on_fine_grid": evaluate(fit, fine, events)["point_log_likelihood"],
                                "fine_ll_on_fine_grid": finer.log_likelihood,
                                "map_spearman": float(spearmanr(lcoarse, lfine).statistic),
                                "top10_area_jaccard": float(area[selc & selfine].sum() / area[selc | selfine].sum())})
            predictions[name + "_intensity"] = np.exp(lcoarse)
            if PROXY in features:
                predictions[name + "_coverage_standardized_relative_intensity"] = np.exp(log_intensity(fit, fine, True))
            logging.info("%s | fine-grid mass error %.2f%% | rank correlation %.4f", name,
                         100 * (coarse_mass / len(events) - 1), convergence[-1]["map_spearman"])
        (out / "fitted_models.json").write_text(json.dumps(fit_parameters, indent=2), encoding="utf-8")
        # External evaluation happens after every training-only CV/integration
        # experiment, with no selection/retraining decision based on its scores.
        ext_rows, ext_scores = [], []
        for name, fit in fitted.items():
            ext_rows.append({"model": name, **evaluate(fit, fine, external)})
            scores = external[["DEPOSIT", "longitude", "latitude", "paper_star_id"]].copy()
            scores["model"] = name
            scores["log_intensity"] = log_intensity(fit, external)
            scores["coverage_standardized_log_intensity"] = log_intensity(fit, external, True)
            bg = log_intensity(fit, fine)
            scores["area_percentile"] = [np.average(bg <= s, weights=fine.area_km2) for s in scores.log_intensity]
            ext_scores.append(scores)
        ext_rows, convergence = pd.DataFrame(ext_rows), pd.DataFrame(convergence)
        audit = coverage_audit(fine, events, external)
        for filename, frame in (("external_validation_summary", ext_rows),
                                ("external_deposit_scores", pd.concat(ext_scores)),
                                ("quadrature_convergence", convergence), ("ppp_coefficients", pd.concat(coeffs)),
                                ("sampling_coverage_audit", audit), ("predictions_5km", predictions)):
            frame.to_csv(out / f"{filename}.csv", index=False, encoding="utf-8-sig")
    write_report(out, summary, ext_rows, convergence, audit, boundary, args)
    logging.info("Completed. %s", out / "README_results.md")


if __name__ == "__main__":
    main()
