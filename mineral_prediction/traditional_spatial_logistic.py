"""Traditional spatial-statistical baseline for the paper's 44/16 deposit split.

The response is deposit presence (1) versus sampled background (0). Copper and
the other geological measurements are predictors. The 16 paper validation
deposits are never used to fit the model or choose its threshold.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Config:
    point_dir: Path = Path("output/paper_point_matching")
    feature_path: Path = Path("output/acawlr_gravity_fault_maps/selected_training_samples_final.csv")
    output_dir: Path = Path("output/paper_traditional_statistics")
    negative_ratio: int = 5
    exclusion_km: float = 2.0
    idw_neighbors: int = 16
    idw_power: float = 2.0
    idw_floor_km: float = 1.0
    random_state: int = 42
    sensitivity_repeats: int = 100


RAW_FEATURES = [
    "Cu_ppm",
    "Au_ppm",
    "Mo_ppm",
    "Fe_pct",
    "dist_fault",
    "fault_density",
    "local_fault_confidence",
    "gravity_bouguer",
]

MODEL_FEATURES = [
    "log_Cu_ppm",
    "log_Au_ppm",
    "log_Mo_ppm",
    "Fe_pct",
    "log_dist_fault",
    "log_fault_density",
    "local_fault_confidence",
    "gravity_bouguer",
]

SPATIAL_TERMS = ["lon_centered", "lat_centered", "lon_sq", "lat_sq", "lon_lat"]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Paper-aligned traditional logistic and spatial-trend logistic analysis."
    )
    parser.add_argument("--point-dir", type=Path, default=Config.point_dir)
    parser.add_argument("--feature-path", type=Path, default=Config.feature_path)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--negative-ratio", type=int, default=Config.negative_ratio)
    parser.add_argument("--exclusion-km", type=float, default=Config.exclusion_km)
    parser.add_argument("--idw-neighbors", type=int, default=Config.idw_neighbors)
    parser.add_argument("--sensitivity-repeats", type=int, default=Config.sensitivity_repeats)
    parser.add_argument("--seed", type=int, default=Config.random_state)
    args = parser.parse_args()
    return Config(
        point_dir=args.point_dir,
        feature_path=args.feature_path,
        output_dir=args.output_dir,
        negative_ratio=args.negative_ratio,
        exclusion_km=args.exclusion_km,
        idw_neighbors=args.idw_neighbors,
        random_state=args.seed,
        sensitivity_repeats=args.sensitivity_repeats,
    )


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def load_inputs(cfg: Config):
    train_path = cfg.point_dir / "paper_figure2g_training_44_model_ready.csv"
    validation_path = cfg.point_dir / "paper_figure2g_validation_16_model_ready.csv"
    for path in (train_path, validation_path, cfg.feature_path):
        if not path.exists():
            raise FileNotFoundError(path)

    train_deposits = pd.read_csv(train_path)
    validation_deposits = pd.read_csv(validation_path)
    sources = pd.read_csv(cfg.feature_path)
    require_columns(train_deposits, ["DEPOSIT", "LONGITUDE", "LATITUDE"], "Training deposits")
    require_columns(validation_deposits, ["DEPOSIT", "LONGITUDE", "LATITUDE"], "Validation deposits")
    require_columns(sources, ["longitude", "latitude", *RAW_FEATURES], "Feature source")
    if len(train_deposits) != 44 or len(validation_deposits) != 16:
        raise ValueError(
            f"Expected the paper split 44/16, got {len(train_deposits)}/{len(validation_deposits)}."
        )

    for frame, columns in (
        (train_deposits, ["LONGITUDE", "LATITUDE"]),
        (validation_deposits, ["LONGITUDE", "LATITUDE"]),
        (sources, ["longitude", "latitude", *RAW_FEATURES]),
    ):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    sources = sources.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)
    return train_deposits, validation_deposits, sources


def nearest_deposit_distance_km(points: pd.DataFrame, deposits: pd.DataFrame) -> np.ndarray:
    tree = BallTree(
        np.radians(deposits[["LATITUDE", "LONGITUDE"]].to_numpy(float)),
        metric="haversine",
    )
    distance, _ = tree.query(
        np.radians(points[["latitude", "longitude"]].to_numpy(float)), k=1
    )
    return distance[:, 0] * EARTH_RADIUS_KM


def idw_align(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Align all predictors to target coordinates with the same IDW operator."""
    source_coords = np.radians(sources[["latitude", "longitude"]].to_numpy(float))
    target_coords = np.radians(targets[["latitude", "longitude"]].to_numpy(float))
    k = min(cfg.idw_neighbors, len(sources))
    tree = BallTree(source_coords, metric="haversine")
    distance_rad, indices = tree.query(target_coords, k=k)
    distance_km = distance_rad * EARTH_RADIUS_KM
    weights = 1.0 / np.maximum(distance_km, cfg.idw_floor_km) ** cfg.idw_power

    aligned = targets.copy().reset_index(drop=True)
    for feature in RAW_FEATURES:
        values = sources[feature].to_numpy(float)[indices]
        valid = np.isfinite(values)
        weighted_sum = np.where(valid, values * weights, 0.0).sum(axis=1)
        weight_sum = np.where(valid, weights, 0.0).sum(axis=1)
        aligned[feature] = np.divide(
            weighted_sum,
            weight_sum,
            out=np.full(len(aligned), np.nan),
            where=weight_sum > 0,
        )
    aligned["nearest_feature_km"] = distance_km[:, 0]
    aligned["mean_feature_neighbor_km"] = distance_km.mean(axis=1)
    return aligned


def make_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_Cu_ppm"] = np.log1p(out["Cu_ppm"].clip(lower=0))
    out["log_Au_ppm"] = np.log1p(out["Au_ppm"].clip(lower=0))
    out["log_Mo_ppm"] = np.log1p(out["Mo_ppm"].clip(lower=0))
    out["log_dist_fault"] = np.log1p(out["dist_fault"].clip(lower=0))
    out["log_fault_density"] = np.log1p(out["fault_density"].clip(lower=0) * 1_000_000)
    # Center before constructing the trend surface to avoid avoidable
    # collinearity between the linear and quadratic coordinate terms.
    out["lon_centered"] = out["longitude"] + 113.5
    out["lat_centered"] = out["latitude"] - 40.0
    out["lon_sq"] = out["lon_centered"] ** 2
    out["lat_sq"] = out["lat_centered"] ** 2
    out["lon_lat"] = out["lon_centered"] * out["lat_centered"]
    return out


def deposit_targets(deposits: pd.DataFrame, split: str) -> pd.DataFrame:
    out = deposits[["DEPOSIT", "LONGITUDE", "LATITUDE", "paper_star_id"]].copy()
    out = out.rename(columns={"LONGITUDE": "longitude", "LATITUDE": "latitude"})
    out["Label"] = 1
    out["paper_split"] = split
    out["sample_role"] = "paper_deposit"
    return out


def choose_background_pools(
    sources: pd.DataFrame,
    all_deposits: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    pool = sources[["longitude", "latitude"]].copy()
    pool["nearest_paper_deposit_km"] = nearest_deposit_distance_km(pool, all_deposits)
    pool = pool[pool["nearest_paper_deposit_km"] > cfg.exclusion_km].copy()
    pool = pool.drop_duplicates(subset=["longitude", "latitude"]).reset_index(drop=True)
    if len(pool) < (44 + 16) * cfg.negative_ratio:
        raise ValueError("Not enough background points after the deposit exclusion buffer.")
    return pool


def sample_background(
    pool: pd.DataFrame,
    cfg: Config,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_train = 44 * cfg.negative_ratio
    n_validation = 16 * cfg.negative_ratio
    selected = rng.choice(len(pool), size=n_train + n_validation, replace=False)
    train_bg = pool.iloc[selected[:n_train]][["longitude", "latitude"]].copy()
    validation_bg = pool.iloc[selected[n_train:]][["longitude", "latitude"]].copy()
    for frame, split in ((train_bg, "training"), (validation_bg, "validation")):
        frame["DEPOSIT"] = ""
        frame["paper_star_id"] = np.nan
        frame["Label"] = 0
        frame["paper_split"] = split
        frame["sample_role"] = "background"
    return train_bg, validation_bg


def build_analysis_tables(cfg: Config):
    train_deposits, validation_deposits, sources = load_inputs(cfg)
    all_deposits = pd.concat([train_deposits, validation_deposits], ignore_index=True)
    background_pool = choose_background_pools(sources, all_deposits, cfg)
    train_bg, validation_bg = sample_background(background_pool, cfg, cfg.random_state)

    train_targets = pd.concat(
        [deposit_targets(train_deposits, "training"), train_bg], ignore_index=True
    )
    validation_targets = pd.concat(
        [deposit_targets(validation_deposits, "validation"), validation_bg], ignore_index=True
    )
    all_targets = pd.concat([train_targets, validation_targets], ignore_index=True)
    aligned = make_predictors(idw_align(all_targets, sources, cfg))
    train = aligned[aligned["paper_split"].eq("training")].reset_index(drop=True)
    validation = aligned[aligned["paper_split"].eq("validation")].reset_index(drop=True)
    return train, validation, background_pool, sources, train_deposits, validation_deposits


def make_model(feature_names: list[str]):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=42),
    )


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    seed: int,
):
    x_train = train[features]
    y_train = train["Label"].to_numpy(int)
    x_validation = validation[features]
    y_validation = validation["Label"].to_numpy(int)
    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians)
    x_validation = x_validation.fillna(medians)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probability = cross_val_predict(
        make_model(features), x_train, y_train, cv=folds, method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(y_train, oof_probability)
    model = make_model(features)
    model.fit(x_train, y_train)
    train_probability = model.predict_proba(x_train)[:, 1]
    validation_probability = model.predict_proba(x_validation)[:, 1]
    return model, medians, threshold, train_probability, validation_probability, oof_probability


def compute_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "brier": brier_score_loss(y_true, probability),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    seed: int,
    repeats: int = 1000,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    values = {name: [] for name in ("auc", "pr_auc", "recall", "f1", "brier")}
    for _ in range(repeats):
        sample = np.concatenate(
            [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        )
        metrics = compute_metrics(y_true[sample], probability[sample], threshold)
        for name in values:
            values[name].append(metrics[name])
    return {
        name: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
        for name, value in values.items()
    }


def statsmodels_coefficients(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = train[features].copy()
    x = x.fillna(x.median(numeric_only=True))
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=features, index=x.index)
    design = sm.add_constant(x_scaled, has_constant="add")
    result = sm.GLM(train["Label"], design, family=sm.families.Binomial()).fit(cov_type="HC3")
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": result.params.values,
            "std_error_hc3": result.bse.values,
            "z": result.tvalues.values,
            "p_value": result.pvalues.values,
            "odds_ratio_per_sd": np.exp(result.params.values),
            "odds_ratio_ci_low": np.exp(ci[0].to_numpy()),
            "odds_ratio_ci_high": np.exp(ci[1].to_numpy()),
        }
    )
    table.attrs["aic"] = float(result.aic)
    table.attrs["bic_deviance"] = float(result.bic_llf)
    return table


def morans_i(residuals: np.ndarray, coordinates: np.ndarray, seed: int, permutations: int = 999):
    n_neighbors = min(8, len(coordinates) - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    indices = nn.kneighbors(return_distance=False)[:, 1:]
    weights = np.zeros((len(coordinates), len(coordinates)), dtype=float)
    rows = np.repeat(np.arange(len(coordinates)), n_neighbors)
    weights[rows, indices.ravel()] = 1.0
    weights = np.maximum(weights, weights.T)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)

    centered = residuals - residuals.mean()
    denominator = np.sum(centered**2)

    def statistic(values):
        c = values - values.mean()
        return len(values) / weights.sum() * np.sum(weights * np.outer(c, c)) / np.sum(c**2)

    observed = float(statistic(residuals))
    rng = np.random.default_rng(seed)
    null = np.array([statistic(rng.permutation(residuals)) for _ in range(permutations)])
    p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1))
    return {"morans_i": observed, "permutation_p_value": p_value, "expected_i": -1 / (len(residuals) - 1)}


def sensitivity_analysis(
    cfg: Config,
    background_pool: pd.DataFrame,
    sources: pd.DataFrame,
    train_deposits: pd.DataFrame,
    validation_deposits: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    train_positive = make_predictors(
        idw_align(deposit_targets(train_deposits, "training"), sources, cfg)
    )
    validation_positive = make_predictors(
        idw_align(deposit_targets(validation_deposits, "validation"), sources, cfg)
    )
    aligned_pool = make_predictors(
        idw_align(background_pool[["longitude", "latitude"]], sources, cfg)
    )
    n_train = 44 * cfg.negative_ratio
    n_validation = 16 * cfg.negative_ratio
    for repeat in range(cfg.sensitivity_repeats):
        seed = cfg.random_state + 1000 + repeat
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(aligned_pool), size=n_train + n_validation, replace=False)
        train_bg = aligned_pool.iloc[selected[:n_train]].copy()
        validation_bg = aligned_pool.iloc[selected[n_train:]].copy()
        for frame, split in ((train_bg, "training"), (validation_bg, "validation")):
            frame["DEPOSIT"] = ""
            frame["paper_star_id"] = np.nan
            frame["Label"] = 0
            frame["paper_split"] = split
            frame["sample_role"] = "background"
        train = pd.concat([train_positive, train_bg], ignore_index=True)
        validation = pd.concat([validation_positive, validation_bg], ignore_index=True)
        for model_name, features in (
            ("global_logistic", MODEL_FEATURES),
            ("spatial_trend_logistic", MODEL_FEATURES + SPATIAL_TERMS),
        ):
            _, _, threshold, _, probability, _ = fit_predict(train, validation, features, seed)
            metrics = compute_metrics(validation["Label"].to_numpy(int), probability, threshold)
            rows.append({"repeat": repeat + 1, "seed": seed, "model": model_name, **metrics})
    return pd.DataFrame(rows)


def plot_results(
    output_path: Path,
    validation: pd.DataFrame,
    results: dict,
    coefficient_table: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    colors = {"global_logistic": "#1f77b4", "spatial_trend_logistic": "#d1495b"}
    y = validation["Label"].to_numpy(int)

    for name, item in results.items():
        fpr, tpr, _ = roc_curve(y, item["validation_probability"])
        precision, recall, _ = precision_recall_curve(y, item["validation_probability"])
        axes[0, 0].plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC={item['metrics']['auc']:.3f})")
        axes[0, 1].plot(recall, precision, lw=2, color=colors[name], label=f"{name} (AP={item['metrics']['pr_auc']:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], color="#777777", ls="--", lw=1)
    axes[0, 0].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC: 16 paper validation deposits")
    axes[0, 1].axhline(y.mean(), color="#777777", ls="--", lw=1)
    axes[0, 1].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)

    map_ax = axes[1, 0]
    background = validation[validation["Label"].eq(0)]
    deposits = validation[validation["Label"].eq(1)]
    probability = results["global_logistic"]["validation_probability"]
    map_ax.scatter(
        background["longitude"], background["latitude"], c=probability[background.index],
        cmap="viridis", vmin=0, vmax=1, s=25, alpha=0.75, label="Validation background"
    )
    scatter = map_ax.scatter(
        deposits["longitude"], deposits["latitude"], c=probability[deposits.index],
        cmap="viridis", vmin=0, vmax=1, s=95, marker="*", edgecolor="black", linewidth=0.6,
        label="16 validation deposits"
    )
    fig.colorbar(scatter, ax=map_ax, label="Predicted relative prospectivity")
    map_ax.set(xlabel="Longitude", ylabel="Latitude", title="Independent validation predictions")
    map_ax.legend(frameon=False, loc="lower left")

    coef = coefficient_table[coefficient_table["term"].ne("const")].copy().iloc[::-1]
    ax = axes[1, 1]
    y_pos = np.arange(len(coef))
    ax.errorbar(
        coef["odds_ratio_per_sd"], y_pos,
        xerr=[coef["odds_ratio_per_sd"] - coef["odds_ratio_ci_low"], coef["odds_ratio_ci_high"] - coef["odds_ratio_per_sd"]],
        fmt="o", color="#2a6f97", ecolor="#7a9e9f", capsize=3
    )
    ax.axvline(1.0, color="#777777", ls="--", lw=1)
    ax.set_yticks(y_pos, coef["term"])
    ax.set_xscale("log")
    ax.set(xlabel="Odds ratio per training SD (95% CI)", title="Global logistic coefficients")
    fig.suptitle("Traditional statistical baseline: deposit presence is the response", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    cfg: Config,
    results: dict,
    intervals: dict,
    moran: dict,
    sensitivity: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    lines = [
        "# 传统空间统计基线分析",
        "",
        "## 分析口径",
        "",
        "- 因变量：矿床存在（deposit presence，1）与背景位置（0）。",
        "- 自变量：Cu、Au、Mo、Fe、断层距离、断层密度、局部断层置信度、Bouguer 重力异常。",
        "- 论文划分：44 个训练矿床参与拟合；16 个验证矿床完全不参与拟合或阈值选择。",
        f"- 背景样本：每个矿床配 {cfg.negative_ratio} 个背景点，并排除距 60 个论文矿床 {cfg.exclusion_km:g} km 内的位置。",
        f"- 空间对齐：所有正负样本均用相同的 {cfg.idw_neighbors} 邻点 IDW 算子提取特征。",
        "- 模型概率解释为相对找矿有利度，不解释为真实自然发生概率。",
        "",
        "## 独立验证结果",
        "",
        "| 模型 | AUC | PR-AUC | 准确率 | 精确率 | 召回率 | F1 | Brier | 阈值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("global_logistic", "spatial_trend_logistic"):
        m = results[name]["metrics"]
        lines.append(
            f"| {name} | {m['auc']:.3f} | {m['pr_auc']:.3f} | {m['accuracy']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['brier']:.3f} | {m['threshold']:.3f} |"
        )
    global_metrics = results["global_logistic"]["metrics"]
    lines += [
        "",
        "## 不确定性与诊断",
        "",
        f"- 全局 Logistic 的 AUC 95% bootstrap 区间：{intervals['auc'][0]:.3f}-{intervals['auc'][1]:.3f}。",
        f"- 全局 Logistic 的召回率 95% bootstrap 区间：{intervals['recall'][0]:.3f}-{intervals['recall'][1]:.3f}。",
        f"- 验证残差 Moran's I = {moran['morans_i']:.3f}，置换检验 p = {moran['permutation_p_value']:.3f}。",
        f"- 固定验证集混淆矩阵：TP={global_metrics['tp']}，FN={global_metrics['fn']}，FP={global_metrics['fp']}，TN={global_metrics['tn']}。",
        "",
        "## 背景抽样敏感性",
        "",
    ]
    grouped = sensitivity.groupby("model")[["auc", "pr_auc", "recall", "f1"]].agg(["mean", "std"])
    for model_name in grouped.index:
        row = grouped.loc[model_name]
        lines.append(
            f"- {model_name}: AUC {row[('auc', 'mean')]:.3f} +/- {row[('auc', 'std')]:.3f}; "
            f"召回率 {row[('recall', 'mean')]:.3f} +/- {row[('recall', 'std')]:.3f}; "
            f"F1 {row[('f1', 'mean')]:.3f} +/- {row[('f1', 'std')]:.3f}。"
        )
    nearest = validation.loc[validation["Label"].eq(1), "nearest_feature_km"]
    lines += [
        "",
        "## 重要限制",
        "",
        f"16 个验证矿床到最近特征源点的距离中位数为 {nearest.median():.1f} km，最大值为 {nearest.max():.1f} km；距离较大的矿床预测不确定性更高。",
        "论文未公开西部美国逐像元训练标签与完整 1 km 特征栅格，因此本结果是可复现的传统病例-对照基线，不是对论文数值的逐点复刻。",
    ]
    (cfg.output_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, background_pool, sources, train_deposits, validation_deposits = build_analysis_tables(cfg)

    results = {}
    coefficient_tables = {}
    model_definitions = {
        "global_logistic": MODEL_FEATURES,
        "spatial_trend_logistic": MODEL_FEATURES + SPATIAL_TERMS,
    }
    for name, features in model_definitions.items():
        model, medians, threshold, train_probability, validation_probability, oof_probability = fit_predict(
            train, validation, features, cfg.random_state
        )
        metrics = compute_metrics(validation["Label"].to_numpy(int), validation_probability, threshold)
        results[name] = {
            "model": model,
            "medians": medians,
            "threshold": threshold,
            "train_probability": train_probability,
            "validation_probability": validation_probability,
            "oof_probability": oof_probability,
            "metrics": metrics,
        }
        coefficient_tables[name] = statsmodels_coefficients(train, features)

    metrics_rows = []
    for name, result in results.items():
        row = {"model": name, **result["metrics"]}
        row["aic"] = coefficient_tables[name].attrs["aic"]
        row["bic_deviance"] = coefficient_tables[name].attrs["bic_deviance"]
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(cfg.output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")

    for name, table in coefficient_tables.items():
        table.to_csv(cfg.output_dir / f"{name}_coefficients.csv", index=False, encoding="utf-8-sig")

    train_output = train.copy()
    validation_output = validation.copy()
    for name, result in results.items():
        train_output[f"{name}_probability"] = result["train_probability"]
        validation_output[f"{name}_probability"] = result["validation_probability"]
        validation_output[f"{name}_predicted"] = (
            result["validation_probability"] >= result["threshold"]
        ).astype(int)
    train_output.to_csv(cfg.output_dir / "training_case_control_data.csv", index=False, encoding="utf-8-sig")
    validation_output.to_csv(cfg.output_dir / "validation_case_control_predictions.csv", index=False, encoding="utf-8-sig")
    validation_output[validation_output["Label"].eq(1)].to_csv(
        cfg.output_dir / "validation_16_deposit_predictions.csv", index=False, encoding="utf-8-sig"
    )

    global_result = results["global_logistic"]
    intervals = bootstrap_metric_intervals(
        validation["Label"].to_numpy(int),
        global_result["validation_probability"],
        global_result["threshold"],
        cfg.random_state,
    )
    residuals = validation["Label"].to_numpy(float) - global_result["validation_probability"]
    moran = morans_i(
        residuals,
        validation[["longitude", "latitude"]].to_numpy(float),
        cfg.random_state,
    )

    sensitivity = sensitivity_analysis(
        cfg, background_pool, sources, train_deposits, validation_deposits
    )
    sensitivity.to_csv(cfg.output_dir / "background_sampling_sensitivity.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary = (
        sensitivity.groupby("model")[["auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier"]]
        .agg(["mean", "std", "min", "max"])
    )
    sensitivity_summary.to_csv(cfg.output_dir / "background_sampling_sensitivity_summary.csv", encoding="utf-8-sig")

    diagnostics = {
        "config": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in cfg.__dict__.items()},
            "response": "deposit presence (1) versus sampled background (0)",
            "predictors": MODEL_FEATURES,
        },
        "sample_counts": {
            "training_deposits": 44,
            "training_background": int((train["Label"] == 0).sum()),
            "validation_deposits": 16,
            "validation_background": int((validation["Label"] == 0).sum()),
        },
        "bootstrap_95_percent_intervals": intervals,
        "validation_residual_moran": moran,
    }
    (cfg.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(
        cfg.output_dir / "traditional_statistical_dashboard.png",
        validation,
        results,
        coefficient_tables["global_logistic"],
    )
    write_summary(cfg, results, intervals, moran, sensitivity, validation)

    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print(f"\nResults written to: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
