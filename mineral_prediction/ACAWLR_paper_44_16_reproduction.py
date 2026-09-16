"""Paper-oriented ACAWLR reproduction using the recovered 44/16 split.

This script keeps the paper's response/predictor roles explicit:

* response: deposit presence on a 1 km modeling grid;
* predictors: Cu, Au, Mo, Fe, fault and Bouguer-gravity features;
* training labels: only the 44 paper training deposits;
* external validation labels: only the 16 paper validation deposits.

The implementation reuses the tested ACAWLR architecture and feature routines
from ``ACAWLR_improved.py``. It does not modify that file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pyproj import CRS, Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ReproConfig:
    core_path: Path = Path(r"D:\code\ResearchPractice\scripts\ACAWLR_improved.py")
    geochemistry_path: Path = Path(r"D:\code\ResearchPractice\data\geochemistry\沉积物地球化学数据.csv")
    fault_path: Path = Path(r"D:\code\ResearchPractice\data\faults\GeologyFaults_USCanada.shp")
    gravity_path: Path = Path(r"D:\code\ResearchPractice\data\gravity\GRIDS\grid_xyz\bouguer.xyz.gz")
    paper_point_dir: Path = Path(r"D:\code\ResearchPractice\mineral_prediction\output\paper_point_matching")
    output_dir: Path = Path(r"D:\code\ResearchPractice\outputs\acawlr_paper_44_16_reproduction_outputs")

    study_bounds: tuple[float, float, float, float] = (-125.0, -102.0, 30.0, 50.0)
    projected_crs: str = "ESRI:102008"
    grid_size_m: float = 1000.0
    positive_buffer_km: float = 2.0
    negative_positive_ratio: int = 20
    max_train_negatives: int = 20000
    max_validation_negatives: int = 8000
    idw_neighbors: int = 12
    idw_power: float = 2.0
    idw_floor_m: float = 1000.0

    base_features: tuple[str, ...] = ("Cu_ppm", "Au_ppm", "Mo_ppm", "Fe_pct")
    model_features: tuple[str, ...] = (
        "Cu_ppm",
        "Au_ppm",
        "Mo_ppm",
        "Fe_pct",
        "dist_fault",
        "fault_density",
        "local_fault_confidence",
        "gravity_bouguer",
        "gravity_distance_deg",
        "has_gravity",
    )

    fault_search_radius_m: float = 50_000.0
    fault_max_near_lines: int = 80
    gravity_max_distance_deg: float = 0.25
    aniso_parallel_scale: float = 3.0
    aniso_perp_scale: float = 1.0
    distance_normalize_m: float = 50_000.0
    gaspg_h: int = 120
    gaspg_w: int = 100

    batch_size: int = 32
    epochs: int = 25
    early_stop_patience: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


EARTH_RADIUS_KM = 6371.0088


def parse_args() -> ReproConfig:
    defaults = ReproConfig()
    parser = argparse.ArgumentParser(
        description="Reproduce ACAWLR with the paper's 44 training and 16 external validation deposits."
    )
    parser.add_argument("--core-path", type=Path, default=defaults.core_path)
    parser.add_argument("--geochemistry-path", type=Path, default=defaults.geochemistry_path)
    parser.add_argument("--fault-path", type=Path, default=defaults.fault_path)
    parser.add_argument("--gravity-path", type=Path, default=defaults.gravity_path)
    parser.add_argument("--paper-point-dir", type=Path, default=defaults.paper_point_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--negative-positive-ratio", type=int, default=defaults.negative_positive_ratio)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--gaspg-h", type=int, default=defaults.gaspg_h)
    parser.add_argument("--gaspg-w", type=int, default=defaults.gaspg_w)
    parser.add_argument("--seed", type=int, default=defaults.random_state)
    args = parser.parse_args()
    return ReproConfig(
        core_path=args.core_path,
        geochemistry_path=args.geochemistry_path,
        fault_path=args.fault_path,
        gravity_path=args.gravity_path,
        paper_point_dir=args.paper_point_dir,
        output_dir=args.output_dir,
        negative_positive_ratio=args.negative_positive_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gaspg_h=args.gaspg_h,
        gaspg_w=args.gaspg_w,
        random_state=args.seed,
    )


def load_core(path: Path):
    if not path.exists():
        local_fallback = Path(__file__).with_name("ACAWLR_gravity_fault_maps.py")
        if local_fallback.exists():
            path = local_fallback
        else:
            raise FileNotFoundError(f"ACAWLR core not found: {path}")
    spec = importlib.util.spec_from_file_location("acawlr_reproduction_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ACAWLR core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        ],
        force=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_files(cfg: ReproConfig) -> None:
    paths = [
        cfg.geochemistry_path,
        cfg.fault_path,
        cfg.gravity_path,
        cfg.paper_point_dir / "paper_figure2g_training_44_model_ready.csv",
        cfg.paper_point_dir / "paper_figure2g_validation_16_model_ready.csv",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def transformers(cfg: ReproConfig):
    geographic = CRS.from_epsg(4326)
    projected = CRS.from_user_input(cfg.projected_crs)
    return (
        Transformer.from_crs(geographic, projected, always_xy=True),
        Transformer.from_crs(projected, geographic, always_xy=True),
    )


def load_paper_deposits(cfg: ReproConfig):
    train = pd.read_csv(cfg.paper_point_dir / "paper_figure2g_training_44_model_ready.csv")
    validation = pd.read_csv(cfg.paper_point_dir / "paper_figure2g_validation_16_model_ready.csv")
    if len(train) != 44 or len(validation) != 16:
        raise ValueError(f"Expected 44/16 paper deposits, found {len(train)}/{len(validation)}")
    needed = {"DEPOSIT", "LONGITUDE", "LATITUDE", "paper_star_id"}
    for name, frame in (("training", train), ("validation", validation)):
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"{name} deposits missing columns: {sorted(missing)}")
        frame["LONGITUDE"] = pd.to_numeric(frame["LONGITUDE"], errors="coerce")
        frame["LATITUDE"] = pd.to_numeric(frame["LATITUDE"], errors="coerce")
        if frame[["LONGITUDE", "LATITUDE"]].isna().any().any():
            raise ValueError(f"{name} deposit coordinates contain missing values")
    train = train.copy()
    validation = validation.copy()
    train["paper_split"] = "training"
    validation["paper_split"] = "validation"
    return train, validation


def add_projected_coordinates(df: pd.DataFrame, forward: Transformer) -> pd.DataFrame:
    out = df.copy()
    x, y = forward.transform(out["longitude"].to_numpy(float), out["latitude"].to_numpy(float))
    out["metric_x"] = np.asarray(x, dtype=float)
    out["metric_y"] = np.asarray(y, dtype=float)
    return out


def positive_grid_cells(
    deposits: pd.DataFrame,
    split: str,
    cfg: ReproConfig,
    forward: Transformer,
    inverse: Transformer,
) -> pd.DataFrame:
    radius_m = cfg.positive_buffer_km * 1000.0
    steps = np.arange(-math.ceil(radius_m / cfg.grid_size_m), math.ceil(radius_m / cfg.grid_size_m) + 1)
    offsets = np.array(
        [
            (dx * cfg.grid_size_m, dy * cfg.grid_size_m)
            for dx in steps
            for dy in steps
            if math.hypot(dx * cfg.grid_size_m, dy * cfg.grid_size_m) <= radius_m + 1e-9
        ],
        dtype=float,
    )
    rows = []
    for deposit in deposits.itertuples(index=False):
        center_x, center_y = forward.transform(float(deposit.LONGITUDE), float(deposit.LATITUDE))
        center_x = round(center_x / cfg.grid_size_m) * cfg.grid_size_m
        center_y = round(center_y / cfg.grid_size_m) * cfg.grid_size_m
        for dx, dy in offsets:
            rows.append(
                {
                    "metric_x": center_x + dx,
                    "metric_y": center_y + dy,
                    "Label": 1,
                    "paper_split": split,
                    "sample_role": "deposit_buffer_grid",
                    "source_deposit": str(deposit.DEPOSIT),
                    "paper_star_id": int(deposit.paper_star_id),
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(["paper_star_id", "metric_x", "metric_y"]).drop_duplicates(
        ["metric_x", "metric_y"], keep="first"
    )
    lon, lat = inverse.transform(out["metric_x"].to_numpy(), out["metric_y"].to_numpy())
    out["longitude"] = lon
    out["latitude"] = lat
    west, east, south, north = cfg.study_bounds
    out = out[out["longitude"].between(west, east) & out["latitude"].between(south, north)]
    return out.reset_index(drop=True)


def projected_study_extent(cfg: ReproConfig, forward: Transformer):
    west, east, south, north = cfg.study_bounds
    perimeter_lon = np.concatenate(
        [
            np.linspace(west, east, 200),
            np.linspace(west, east, 200),
            np.full(200, west),
            np.full(200, east),
        ]
    )
    perimeter_lat = np.concatenate(
        [
            np.full(200, south),
            np.full(200, north),
            np.linspace(south, north, 200),
            np.linspace(south, north, 200),
        ]
    )
    x, y = forward.transform(perimeter_lon, perimeter_lat)
    grid = cfg.grid_size_m
    return (
        math.floor(min(x) / grid) * grid,
        math.ceil(max(x) / grid) * grid,
        math.floor(min(y) / grid) * grid,
        math.ceil(max(y) / grid) * grid,
    )


def sample_background_grid(
    n_needed: int,
    occupied_keys: set[tuple[int, int]],
    cfg: ReproConfig,
    forward: Transformer,
    inverse: Transformer,
    seed: int,
    split: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = projected_study_extent(cfg, forward)
    x_count = int(round((xmax - xmin) / cfg.grid_size_m)) + 1
    y_count = int(round((ymax - ymin) / cfg.grid_size_m)) + 1
    accepted: dict[tuple[int, int], dict] = {}
    west, east, south, north = cfg.study_bounds

    while len(accepted) < n_needed:
        batch_n = max((n_needed - len(accepted)) * 4, 2000)
        ix = rng.integers(0, x_count, size=batch_n)
        iy = rng.integers(0, y_count, size=batch_n)
        metric_x = xmin + ix * cfg.grid_size_m
        metric_y = ymin + iy * cfg.grid_size_m
        lon, lat = inverse.transform(metric_x, metric_y)
        candidates = pd.DataFrame(
            {"metric_x": metric_x, "metric_y": metric_y, "longitude": lon, "latitude": lat}
        )
        candidates = candidates[
            candidates["longitude"].between(west, east)
            & candidates["latitude"].between(south, north)
        ]
        if candidates.empty:
            continue
        for row in candidates.itertuples(index=False):
            key = (int(round(row.metric_x)), int(round(row.metric_y)))
            if key in occupied_keys or key in accepted:
                continue
            accepted[key] = {
                "metric_x": row.metric_x,
                "metric_y": row.metric_y,
                "longitude": row.longitude,
                "latitude": row.latitude,
                "Label": 0,
                "paper_split": split,
                "sample_role": "background_grid",
                "source_deposit": "",
                "paper_star_id": np.nan,
            }
            if len(accepted) >= n_needed:
                break
    occupied_keys.update(accepted.keys())
    return pd.DataFrame(accepted.values())


def build_grid_samples(cfg: ReproConfig, train_deposits: pd.DataFrame, validation_deposits: pd.DataFrame):
    forward, inverse = transformers(cfg)
    train_positive = positive_grid_cells(train_deposits, "training", cfg, forward, inverse)
    validation_positive = positive_grid_cells(validation_deposits, "validation", cfg, forward, inverse)
    all_positive = pd.concat([train_positive, validation_positive], ignore_index=True).drop_duplicates(
        ["metric_x", "metric_y"]
    )
    occupied = {
        (int(round(row.metric_x)), int(round(row.metric_y)))
        for row in all_positive.itertuples(index=False)
    }
    n_train_negative = min(
        len(train_positive) * cfg.negative_positive_ratio,
        cfg.max_train_negatives,
    )
    n_validation_negative = min(
        len(validation_positive) * cfg.negative_positive_ratio,
        cfg.max_validation_negatives,
    )
    train_negative = sample_background_grid(
        n_train_negative,
        occupied,
        cfg,
        forward,
        inverse,
        cfg.random_state,
        "training",
    )
    validation_negative = sample_background_grid(
        n_validation_negative,
        occupied,
        cfg,
        forward,
        inverse,
        cfg.random_state + 1,
        "validation",
    )
    train = pd.concat([train_positive, train_negative], ignore_index=True)
    validation = pd.concat([validation_positive, validation_negative], ignore_index=True)
    train["sample_id"] = np.arange(len(train), dtype=int)
    validation["sample_id"] = np.arange(len(validation), dtype=int) + len(train)
    logging.info(
        "Grid samples: train pos=%s neg=%s; external validation pos=%s neg=%s",
        len(train_positive), len(train_negative), len(validation_positive), len(validation_negative),
    )
    return train, validation


def read_geochemistry(cfg: ReproConfig) -> pd.DataFrame:
    logging.info("Reading geochemistry: %s", cfg.geochemistry_path)
    data = pd.read_csv(cfg.geochemistry_path, low_memory=False)
    needed = ["longitude", "latitude", *cfg.base_features]
    missing = [column for column in needed if column not in data.columns]
    if missing:
        raise ValueError(f"Geochemistry file missing columns: {missing}")
    for column in needed:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    west, east, south, north = cfg.study_bounds
    data = data.dropna(subset=["longitude", "latitude"])
    data = data[
        data["longitude"].between(west, east)
        & data["latitude"].between(south, north)
    ].reset_index(drop=True)
    if data.empty:
        raise ValueError("No geochemistry samples remain inside the study bounds")
    logging.info("Geochemistry points inside study area: %s", len(data))
    return data


def idw_geochemistry(targets: pd.DataFrame, sources: pd.DataFrame, cfg: ReproConfig) -> pd.DataFrame:
    forward, _ = transformers(cfg)
    sx, sy = forward.transform(sources["longitude"].to_numpy(), sources["latitude"].to_numpy())
    source_xy = np.column_stack([sx, sy])
    target_xy = targets[["metric_x", "metric_y"]].to_numpy(float)
    k = min(cfg.idw_neighbors, len(sources))
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(source_xy)
    distances, indices = nn.kneighbors(target_xy, return_distance=True)
    weights = 1.0 / np.maximum(distances, cfg.idw_floor_m) ** cfg.idw_power
    out = targets.copy().reset_index(drop=True)
    for feature in cfg.base_features:
        values = sources[feature].to_numpy(float)[indices]
        valid = np.isfinite(values)
        numerator = np.where(valid, values * weights, 0.0).sum(axis=1)
        denominator = np.where(valid, weights, 0.0).sum(axis=1)
        out[feature] = np.divide(
            numerator,
            denominator,
            out=np.full(len(out), np.nan),
            where=denominator > 0,
        )
    out["nearest_geochemistry_km"] = distances[:, 0] / 1000.0
    out["mean_geochemistry_neighbor_km"] = distances.mean(axis=1) / 1000.0
    return out


def make_core_config(cfg: ReproConfig, core):
    core_cfg = core.Config(
        data_path=cfg.geochemistry_path,
        fault_path=cfg.fault_path,
        deposit_path=cfg.paper_point_dir / "paper_figure2g_training_44_model_ready.csv",
        output_dir=cfg.output_dir,
        study_bounds=cfg.study_bounds,
        deposit_buffer_km=cfg.positive_buffer_km,
        neg_pos_ratio=cfg.negative_positive_ratio,
        max_negatives=cfg.max_train_negatives,
        random_state=cfg.random_state,
        fault_search_radius_m=cfg.fault_search_radius_m,
        fault_max_near_lines=cfg.fault_max_near_lines,
        gravity_grid_path=cfg.gravity_path,
        gravity_max_distance_deg=cfg.gravity_max_distance_deg,
        aniso_parallel_scale=cfg.aniso_parallel_scale,
        aniso_perp_scale=cfg.aniso_perp_scale,
        distance_normalize_m=cfg.distance_normalize_m,
        grid_h=cfg.gaspg_h,
        grid_w=cfg.gaspg_w,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        early_stop_patience=cfg.early_stop_patience,
        device=cfg.device,
        reuse_fault_features=True,
        reuse_gravity_features=True,
        make_maps=False,
    )
    core.CFG = core_cfg
    core.DEVICE = torch.device(cfg.device)
    return core_cfg


def extract_all_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cfg: ReproConfig,
    core,
    core_cfg,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train, validation], ignore_index=True)
    sources = read_geochemistry(cfg)
    combined = idw_geochemistry(combined, sources, cfg)
    logging.info("Computing/reusing fault features for %s grid cells", len(combined))
    combined = core.compute_fault_features(combined, "longitude", "latitude", core_cfg)
    logging.info("Computing/reusing Bouguer gravity features")
    combined = core.compute_gravity_features(combined, "longitude", "latitude", core_cfg)
    combined.to_csv(cfg.output_dir / "all_modeling_grid_features.csv", index=False, encoding="utf-8-sig")
    train_features = combined[combined["paper_split"].eq("training")].reset_index(drop=True)
    validation_features = combined[combined["paper_split"].eq("validation")].reset_index(drop=True)
    return train_features, validation_features


def fit_preprocessor(train: pd.DataFrame, features: list[str]):
    medians = train[features].median(numeric_only=True)
    filled = train[features].fillna(medians)
    means = filled.mean()
    stds = filled.std().replace(0, 1.0)
    return medians, means, stds


def tensorize(df: pd.DataFrame, features: list[str], medians, means, stds):
    values = (df[features].fillna(medians) - means) / stds
    x_raw = torch.tensor(values.to_numpy(float), dtype=torch.float32)
    x = torch.cat([x_raw, torch.ones(len(x_raw), 1)], dim=1)
    y = torch.tensor(df["Label"].to_numpy(float), dtype=torch.float32).view(-1, 1)
    coords = torch.tensor(df[["metric_x", "metric_y"]].to_numpy(float), dtype=torch.float32)
    theta = torch.tensor(df["local_fault_angle"].to_numpy(float), dtype=torch.float32).view(-1, 1)
    return x, y, coords, theta


def global_beta(x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.Tensor:
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=seed,
    )
    classifier.fit(x[:, :-1].numpy(), y.view(-1).numpy().astype(int))
    beta = np.concatenate([classifier.coef_.ravel(), classifier.intercept_.ravel()])
    return torch.tensor(beta, dtype=torch.float32)


def make_model(core, x: torch.Tensor, y: torch.Tensor, cfg: ReproConfig):
    beta = global_beta(x, y, cfg.random_state)
    model = core.ACAWLR(coef_num=x.shape[1], beta_global=beta).to(cfg.device)
    return model, beta


def train_epoch_loop(
    model,
    train_tensors,
    validation_tensors,
    core,
    core_cfg,
    cfg: ReproConfig,
    max_epochs: int,
    early_stop: bool,
):
    x_train, y_train, coords_train, theta_train = [tensor.to(cfg.device) for tensor in train_tensors]
    x_val, y_val, coords_val, theta_val = (
        [tensor.to(cfg.device) for tensor in validation_tensors] if validation_tensors is not None else (None,) * 4
    )
    n_pos = float(y_train.sum().item())
    n_neg = float(y_train.numel() - y_train.sum().item())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / max(n_pos, 1.0)], device=cfg.device)
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.random_state)
    loader = DataLoader(
        TensorDataset(coords_train, theta_train, x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    grid_x, grid_y = core.make_global_grid(
        coords_train, cfg.gaspg_h, cfg.gaspg_w, device=torch.device(cfg.device)
    )
    best_state = None
    best_epoch = max_epochs
    best_ap = -np.inf
    patience = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_coords, batch_theta, batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            aniso = core.generate_anisotropic_grid(grid_x, grid_y, batch_coords, batch_theta, core_cfg)
            logits = model(aniso, batch_x)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            total_count += len(batch_x)
        train_loss = total_loss / max(total_count, 1)
        row = {"epoch": epoch, "train_loss": train_loss}
        if validation_tensors is not None:
            probability, label = core.predict_prob(
                model, grid_x, grid_y, coords_val, theta_val, x_val, core_cfg, y_val
            )
            ap = average_precision_score(label, probability)
            auc = roc_auc_score(label, probability)
            row.update({"internal_val_ap": ap, "internal_val_auc": auc})
            logging.info(
                "Epoch %02d/%s | loss=%.5f | internal AP=%.4f | internal AUC=%.4f",
                epoch, max_epochs, train_loss, ap, auc,
            )
            if ap > best_ap + 1e-6:
                best_ap = ap
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if early_stop and patience >= cfg.early_stop_patience:
                break
        else:
            logging.info("Final fit epoch %02d/%s | loss=%.5f", epoch, max_epochs, train_loss)
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, grid_x, grid_y, best_epoch, pd.DataFrame(history)


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metric_row(name: str, y_true: np.ndarray, probability: np.ndarray, threshold: float):
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "model": name,
        "auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def assign_grouped_fivefold(train: pd.DataFrame, cfg: ReproConfig) -> pd.Series:
    """Assign deposit buffers and spatial background blocks to five folds."""
    rng = np.random.default_rng(cfg.random_state)
    fold = pd.Series(index=train.index, dtype="int64")

    deposit_names = np.array(sorted(train.loc[train["Label"].eq(1), "source_deposit"].unique()))
    rng.shuffle(deposit_names)
    for fold_number, names in enumerate(np.array_split(deposit_names, 5), start=1):
        fold.loc[train["Label"].eq(1) & train["source_deposit"].isin(names)] = fold_number

    negatives = train[train["Label"].eq(0)].copy()
    negatives["spatial_block"] = (
        np.floor(negatives["metric_x"] / 100_000).astype(int).astype(str)
        + "_"
        + np.floor(negatives["metric_y"] / 100_000).astype(int).astype(str)
    )
    block_sizes = negatives.groupby("spatial_block").size().to_dict()
    blocks = list(block_sizes)
    rng.shuffle(blocks)
    blocks.sort(key=lambda block: block_sizes[block], reverse=True)
    fold_load = {number: 0 for number in range(1, 6)}
    block_fold = {}
    for block in blocks:
        target_fold = min(fold_load, key=fold_load.get)
        block_fold[block] = target_fold
        fold_load[target_fold] += block_sizes[block]
    fold.loc[negatives.index] = negatives["spatial_block"].map(block_fold).astype(int)
    if fold.isna().any() or set(fold.astype(int).unique()) != {1, 2, 3, 4, 5}:
        raise RuntimeError("Unable to construct all five grouped folds")
    return fold.astype(int)


def train_models(train: pd.DataFrame, validation: pd.DataFrame, cfg: ReproConfig, core, core_cfg):
    features = list(cfg.model_features)
    train = train.copy()
    train["cv_fold"] = assign_grouped_fivefold(train, cfg)
    oof_acawlr = np.full(len(train), np.nan, dtype=float)
    oof_logistic = np.full(len(train), np.nan, dtype=float)
    fold_metrics = []
    best_epochs = []
    histories = []

    for fold_number in range(1, 6):
        fold_train = train[train["cv_fold"].ne(fold_number)].copy()
        fold_validation = train[train["cv_fold"].eq(fold_number)].copy()
        logging.info(
            "Fivefold %s/5: train=%s (pos=%s), validation=%s (pos=%s)",
            fold_number,
            len(fold_train),
            int(fold_train["Label"].sum()),
            len(fold_validation),
            int(fold_validation["Label"].sum()),
        )
        prep = fit_preprocessor(fold_train, features)
        train_tensors = tensorize(fold_train, features, *prep)
        fold_validation_tensors = tensorize(fold_validation, features, *prep)
        model, beta = make_model(core, train_tensors[0], train_tensors[1], cfg)
        model, grid_x, grid_y, best_epoch, history = train_epoch_loop(
            model,
            train_tensors,
            fold_validation_tensors,
            core,
            core_cfg,
            cfg,
            cfg.epochs,
            early_stop=True,
        )
        best_epochs.append(best_epoch)
        history["stage"] = "fivefold_cross_validation"
        history["fold"] = fold_number
        histories.append(history)

        fold_probability, fold_label = core.predict_prob(
            model,
            grid_x,
            grid_y,
            fold_validation_tensors[2].to(cfg.device),
            fold_validation_tensors[3].to(cfg.device),
            fold_validation_tensors[0].to(cfg.device),
            core_cfg,
            fold_validation_tensors[1].to(cfg.device),
        )
        training_probability, training_label = core.predict_prob(
            model,
            grid_x,
            grid_y,
            train_tensors[2].to(cfg.device),
            train_tensors[3].to(cfg.device),
            train_tensors[0].to(cfg.device),
            core_cfg,
            train_tensors[1].to(cfg.device),
        )
        fold_threshold = choose_threshold(training_label, training_probability)
        oof_acawlr[fold_validation.index] = fold_probability
        acawlr_metrics = metric_row(
            "ACAWLR", fold_label, fold_probability, fold_threshold
        )
        acawlr_metrics.update({"fold": fold_number, "best_epoch": best_epoch})
        fold_metrics.append(acawlr_metrics)

        scaler = StandardScaler()
        fold_train_values = fold_train[features].fillna(prep[0])
        fold_validation_values = fold_validation[features].fillna(prep[0])
        logistic = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=cfg.random_state + fold_number,
        ).fit(scaler.fit_transform(fold_train_values), fold_train["Label"].to_numpy(int))
        logistic_train_probability = logistic.predict_proba(
            scaler.transform(fold_train_values)
        )[:, 1]
        logistic_threshold = choose_threshold(
            fold_train["Label"].to_numpy(int), logistic_train_probability
        )
        logistic_fold_probability = logistic.predict_proba(
            scaler.transform(fold_validation_values)
        )[:, 1]
        oof_logistic[fold_validation.index] = logistic_fold_probability
        logistic_metrics = metric_row(
            "Global logistic",
            fold_validation["Label"].to_numpy(int),
            logistic_fold_probability,
            logistic_threshold,
        )
        logistic_metrics.update({"fold": fold_number, "best_epoch": np.nan})
        fold_metrics.append(logistic_metrics)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "beta_global": beta,
                "feature_columns": features,
                "medians": prep[0].to_dict(),
                "means": prep[1].to_dict(),
                "stds": prep[2].to_dict(),
                "threshold": fold_threshold,
                "fold": fold_number,
                "best_epoch": best_epoch,
            },
            cfg.output_dir / f"acawlr_cv_fold_{fold_number}.pt",
        )

    if not np.isfinite(oof_acawlr).all() or not np.isfinite(oof_logistic).all():
        raise RuntimeError("Fivefold OOF predictions are incomplete")
    acawlr_threshold = choose_threshold(train["Label"].to_numpy(int), oof_acawlr)
    logistic_threshold = choose_threshold(train["Label"].to_numpy(int), oof_logistic)
    final_epoch = max(1, int(round(float(np.median(best_epochs)))))
    logging.info(
        "Fivefold completed. Fold best epochs=%s; final retraining epochs=%s",
        best_epochs,
        final_epoch,
    )

    final_prep = fit_preprocessor(train, features)
    train_tensors = tensorize(train, features, *final_prep)
    validation_tensors = tensorize(validation, features, *final_prep)
    final_model, beta = make_model(core, train_tensors[0], train_tensors[1], cfg)
    final_model, grid_x, grid_y, _, final_history = train_epoch_loop(
        final_model,
        train_tensors,
        None,
        core,
        core_cfg,
        cfg,
        final_epoch,
        early_stop=False,
    )
    acawlr_probability, validation_label = core.predict_prob(
        final_model,
        grid_x,
        grid_y,
        validation_tensors[2].to(cfg.device),
        validation_tensors[3].to(cfg.device),
        validation_tensors[0].to(cfg.device),
        core_cfg,
        validation_tensors[1].to(cfg.device),
    )

    scaler = StandardScaler()
    train_values = train[features].fillna(final_prep[0])
    validation_values = validation[features].fillna(final_prep[0])
    logistic = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=cfg.random_state,
    ).fit(scaler.fit_transform(train_values), train["Label"].to_numpy(int))
    logistic_probability = logistic.predict_proba(scaler.transform(validation_values))[:, 1]

    results = {
        "acawlr": {
            "probability": acawlr_probability,
            "threshold": acawlr_threshold,
            "metrics": metric_row("ACAWLR", validation_label, acawlr_probability, acawlr_threshold),
        },
        "global_logistic": {
            "probability": logistic_probability,
            "threshold": logistic_threshold,
            "metrics": metric_row(
                "Global logistic", validation_label, logistic_probability, logistic_threshold
            ),
        },
    }
    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "beta_global": beta,
        "feature_columns": features,
        "medians": final_prep[0].to_dict(),
        "means": final_prep[1].to_dict(),
        "stds": final_prep[2].to_dict(),
        "threshold": acawlr_threshold,
        "fivefold_best_epochs": best_epochs,
        "final_retraining_epochs": final_epoch,
        "paper_training_deposits": 44,
        "paper_external_validation_deposits": 16,
    }
    torch.save(checkpoint, cfg.output_dir / "acawlr_paper_44_16_model.pt")
    final_history["stage"] = "final_all_44_training_deposits"
    final_history["fold"] = 0
    histories.append(final_history)
    pd.concat(histories, ignore_index=True).to_csv(
        cfg.output_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    oof = train[["sample_id", "Label", "source_deposit", "cv_fold", "longitude", "latitude"]].copy()
    oof["acawlr_oof_probability"] = oof_acawlr
    oof["global_logistic_oof_probability"] = oof_logistic
    return (
        results,
        final_model,
        grid_x,
        grid_y,
        final_prep,
        pd.DataFrame(fold_metrics),
        oof,
    )


def validation_deposit_summary(
    validation: pd.DataFrame,
    validation_deposits: pd.DataFrame,
    results: dict,
) -> pd.DataFrame:
    positive = validation[validation["Label"].eq(1)].copy()
    for name, result in results.items():
        positive[f"{name}_probability"] = result["probability"][validation["Label"].to_numpy() == 1]
    aggregations = {
        "paper_star_id": "first",
        "nearest_geochemistry_km": "median",
    }
    for name in results:
        aggregations[f"{name}_probability"] = "max"
    summary = positive.groupby("source_deposit", as_index=False).agg(aggregations)
    exact_coordinates = validation_deposits[["DEPOSIT", "LONGITUDE", "LATITUDE"]].rename(
        columns={"DEPOSIT": "source_deposit", "LONGITUDE": "longitude", "LATITUDE": "latitude"}
    )
    summary = summary.merge(exact_coordinates, on="source_deposit", how="left", validate="one_to_one")
    for name, result in results.items():
        summary[f"{name}_detected"] = (
            summary[f"{name}_probability"] >= result["threshold"]
        ).astype(int)
    return summary.sort_values("paper_star_id").reset_index(drop=True)


def draw_base_map(ax, cfg: ReproConfig):
    import geopandas as gpd

    gravity = pd.read_csv(
        cfg.gravity_path,
        sep=r"\s+",
        names=["longitude", "latitude", "gravity_bouguer"],
        compression="gzip",
        dtype=np.float32,
    )
    west, east, south, north = cfg.study_bounds
    gravity = gravity[
        gravity["longitude"].between(west, east)
        & gravity["latitude"].between(south, north)
    ]
    step = 0.1
    gravity["lon_bin"] = np.floor((gravity["longitude"] - west) / step) * step + west
    gravity["lat_bin"] = np.floor((gravity["latitude"] - south) / step) * step + south
    raster = gravity.groupby(["lat_bin", "lon_bin"], observed=True)["gravity_bouguer"].mean().unstack()
    image = ax.imshow(
        raster.to_numpy(), origin="lower",
        extent=[raster.columns.min(), raster.columns.max() + step, raster.index.min(), raster.index.max() + step],
        cmap="RdBu_r", alpha=0.82, aspect="auto",
    )
    faults = gpd.read_file(cfg.fault_path)
    if faults.crs is None:
        faults = faults.set_crs("EPSG:4326", allow_override=True)
    else:
        faults = faults.to_crs("EPSG:4326")
    faults = faults.cx[west:east, south:north]
    if len(faults):
        faults.plot(ax=ax, color="#222222", linewidth=0.25, alpha=0.38)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return image


def make_visualizations(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_deposits: pd.DataFrame,
    validation_deposits: pd.DataFrame,
    results: dict,
    deposit_summary: pd.DataFrame,
    cfg: ReproConfig,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    image = draw_base_map(axes[0], cfg)
    axes[0].scatter(
        train.loc[train["Label"].eq(0), "longitude"],
        train.loc[train["Label"].eq(0), "latitude"],
        s=2, c="#777777", alpha=0.12, linewidths=0, label="Training background grids",
    )
    axes[0].scatter(
        train_deposits["LONGITUDE"], train_deposits["LATITUDE"], marker="*", s=42,
        c="#f4a261", edgecolor="black", linewidth=0.25, label="44 training deposits",
    )
    axes[0].scatter(
        validation_deposits["LONGITUDE"], validation_deposits["LATITUDE"], marker="*", s=58,
        c="#00b4d8", edgecolor="black", linewidth=0.35, label="16 external validation deposits",
    )
    axes[0].set_title("A. Paper split and model inputs")
    axes[0].legend(loc="lower left", frameon=True, fontsize=8)
    fig.colorbar(image, ax=axes[0], shrink=0.76, label="Bouguer gravity (mGal)")

    axes[1].scatter(
        validation.loc[validation["Label"].eq(0), "longitude"],
        validation.loc[validation["Label"].eq(0), "latitude"],
        c=results["acawlr"]["probability"][validation["Label"].to_numpy() == 0],
        cmap="magma", vmin=0, vmax=1, s=5, alpha=0.45, linewidths=0,
    )
    deposit_plot = axes[1].scatter(
        deposit_summary["longitude"], deposit_summary["latitude"],
        c=deposit_summary["acawlr_probability"], cmap="magma", vmin=0, vmax=1,
        marker="*", s=100, edgecolor="white", linewidth=0.6,
    )
    axes[1].set_xlim(cfg.study_bounds[0], cfg.study_bounds[1])
    axes[1].set_ylim(cfg.study_bounds[2], cfg.study_bounds[3])
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("B. Strict external-validation probabilities")
    fig.colorbar(deposit_plot, ax=axes[1], shrink=0.76, label="ACAWLR probability")
    fig.suptitle("Paper-oriented ACAWLR reproduction: deposit presence is the response", fontweight="bold")
    fig.savefig(cfg.output_dir / "map_inputs_and_external_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    y = validation["Label"].to_numpy(int)
    colors = {"acawlr": "#d1495b", "global_logistic": "#1f77b4"}
    for name, result in results.items():
        probability = result["probability"]
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        axes[0].plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} AUC={result['metrics']['auc']:.3f}")
        axes[1].plot(recall, precision, color=colors[name], lw=2, label=f"{name} AP={result['metrics']['pr_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#888888", lw=1)
    axes[1].axhline(y.mean(), ls="--", color="#888888", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="External-validation ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="External-validation precision-recall")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.savefig(cfg.output_dir / "external_validation_roc_pr.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    cfg: ReproConfig,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    results: dict,
    deposit_summary: pd.DataFrame,
    cv_metrics: pd.DataFrame,
) -> None:
    metrics = pd.DataFrame([result["metrics"] for result in results.values()])
    detected = {
        name: int(deposit_summary[f"{name}_detected"].sum()) for name in results
    }
    lines = [
        "# ACAWLR 论文 44/16 点正规复现",
        "",
        "## 口径",
        "",
        "- 因变量为 1 km 网格上的矿床存在（1/0）。",
        "- Cu、Au、Mo、Fe、断层与 Bouguer 重力均为自变量。",
        "- 44 个训练矿床的 2 km 缓冲只用于生成训练正样本。",
        "- 16 个验证矿床的 2 km 缓冲只用于严格外部验证，从不参与训练、早停或阈值选择。",
        "- 44 个训练矿床按矿床分组进行五折交叉验证；背景网格按 100 km 空间块分组，避免相邻背景点跨折泄漏。",
        "- 最终轮数取五折最佳轮数的中位数，阈值来自五折 OOF 预测；随后用全部 44 点重新拟合最终模型。",
        "",
        "## 样本",
        "",
        f"- 训练网格：{len(train)}，正样本 {int(train['Label'].sum())}，背景 {int((train['Label'] == 0).sum())}。",
        f"- 外部验证网格：{len(validation)}，正样本 {int(validation['Label'].sum())}，背景 {int((validation['Label'] == 0).sum())}。",
        "",
        "## 外部验证指标",
        "",
        "| 模型 | AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.auc:.3f} | {row.pr_auc:.3f} | {row.accuracy:.3f} | "
            f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.threshold:.3f} |"
        )
    lines += [
        "",
        "## 44 点五折交叉验证",
        "",
        "| 模型 | AUC（均值±标准差） | PR-AUC（均值±标准差） | Recall（均值±标准差） | F1（均值±标准差） |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name, group in cv_metrics.groupby("model", sort=False):
        lines.append(
            f"| {model_name} | {group['auc'].mean():.3f}±{group['auc'].std():.3f} | "
            f"{group['pr_auc'].mean():.3f}±{group['pr_auc'].std():.3f} | "
            f"{group['recall'].mean():.3f}±{group['recall'].std():.3f} | "
            f"{group['f1'].mean():.3f}±{group['f1'].std():.3f} |"
        )
    lines += [
        "",
        "## 16 个验证矿床",
        "",
        f"- ACAWLR 在矿床缓冲内至少命中一个 1 km 网格：{detected['acawlr']}/16。",
        f"- 全局 Logistic 在矿床缓冲内至少命中一个 1 km 网格：{detected['global_logistic']}/16。",
        "",
        "## 可复现性说明",
        "",
        "论文未公开西部美国逐像元训练表和原始随机种子，因此本脚本严格复现公开的方法口径与恢复出的 44/16 空间划分，但不承诺逐位复现论文表 S4 数值。",
    ]
    (cfg.output_dir / "README_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    setup_logging(cfg.output_dir)
    seed_everything(cfg.random_state)
    require_files(cfg)
    core = load_core(cfg.core_path)
    core_cfg = make_core_config(cfg, core)
    logging.info("Device: %s", cfg.device)
    logging.info(
        "Config:\n%s",
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
            ensure_ascii=False,
            indent=2,
        ),
    )

    train_deposits, validation_deposits = load_paper_deposits(cfg)
    train, validation = build_grid_samples(cfg, train_deposits, validation_deposits)

    # This is the central leakage guard: validation-deposit buffers can never
    # appear in the training frame, including as negative samples.
    if set(train["paper_split"].unique()) != {"training"}:
        raise RuntimeError("Validation labels leaked into the training grid")
    train, validation = extract_all_features(train, validation, cfg, core, core_cfg)
    train.to_csv(cfg.output_dir / "training_grid_44_deposits.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(cfg.output_dir / "external_validation_grid_16_deposits.csv", index=False, encoding="utf-8-sig")

    results, _, _, _, _, cv_metrics, oof_predictions = train_models(
        train, validation, cfg, core, core_cfg
    )
    cv_metrics.to_csv(
        cfg.output_dir / "fivefold_cv_metrics.csv", index=False, encoding="utf-8-sig"
    )
    oof_predictions.to_csv(
        cfg.output_dir / "fivefold_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics = pd.DataFrame([result["metrics"] for result in results.values()])
    metrics.to_csv(cfg.output_dir / "external_validation_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_output = validation.copy()
    for name, result in results.items():
        prediction_output[f"{name}_probability"] = result["probability"]
        prediction_output[f"{name}_predicted"] = (
            result["probability"] >= result["threshold"]
        ).astype(int)
    prediction_output.to_csv(
        cfg.output_dir / "external_validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    deposit_summary = validation_deposit_summary(validation, validation_deposits, results)
    if len(deposit_summary) != 16:
        raise RuntimeError(f"Expected predictions for 16 validation deposits, found {len(deposit_summary)}")
    deposit_summary.to_csv(
        cfg.output_dir / "external_validation_16_deposit_summary.csv", index=False, encoding="utf-8-sig"
    )
    make_visualizations(
        train, validation, train_deposits, validation_deposits, results, deposit_summary, cfg
    )
    write_summary(cfg, train, validation, results, deposit_summary, cv_metrics)
    logging.info("External validation metrics:\n%s", metrics.to_string(index=False))
    logging.info("Done. Results: %s", cfg.output_dir.resolve())


if __name__ == "__main__":
    main()
