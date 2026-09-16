"""Presence-only porphyry-copper modelling with lithology and spatial validation.

This analysis is deliberately separate from the existing ACAWLR reproduction.
It implements the two methodological changes motivated by Warton & Shepherd
(2010) and Airola et al. (2019):

* the response is the number of the 44 training deposits in each quadrature
  cell, not a set of correlated positive cells created with a 2 km buffer;
* background cells are equal-area quadrature cells for a Poisson point-process
  likelihood, not verified non-deposit samples;
* USGS GeMS ``Igneous, intrusive`` polygons and distance to their contacts are
  added as geological predictors;
* validation uses whole spatial blocks, optional dead zones, within-fold
  pairwise AUC, a sampled LPO-SCV-style audit, and the untouched 16 deposits.

The output is relative deposit-event intensity.  It is not an absolute
probability of mineralisation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


@dataclass
class Config:
    geochemistry_path: Path = Path(r"D:\code\ResearchPractice\data\geochemistry\沉积物地球化学数据.csv")
    fault_path: Path = Path(r"D:\code\ResearchPractice\data\faults\GeologyFaults_USCanada.shp")
    gravity_path: Path = Path(r"D:\code\ResearchPractice\data\gravity\GRIDS\grid_xyz\bouguer.xyz.gz")
    lithology_path: Path = Path(
        r"D:\code\ResearchPractice\data\lithology\NGMDB_GeMS_4742\GeMS_shapefiles\MapUnitPolys.shp"
    )
    paper_point_dir: Path = Path(
        r"D:\code\ResearchPractice\mineral_prediction\output\paper_point_matching"
    )
    output_dir: Path = Path(
        r"D:\code\ResearchPractice\mineral_prediction\output\ppp_lithology_spatial_validation"
    )

    study_bounds: tuple[float, float, float, float] = (-125.0, -102.0, 30.0, 50.0)
    equal_area_crs: str = "ESRI:102039"
    grid_km: float = 20.0
    block_km: float = 200.0
    buffer_radii_km: tuple[float, ...] = (0.0, 25.0, 50.0)
    n_splits: int = 5
    l2_alpha: float = 0.1
    random_state: int = 42
    lpo_pairs_per_positive: int = 2
    lpo_radius_km: float = 50.0
    skip_lpo: bool = False
    reuse_features: bool = True

    idw_neighbors: int = 12
    idw_power: float = 2.0
    idw_floor_m: float = 1000.0
    gravity_max_distance_deg: float = 0.25


RAW_BASE_FEATURES = [
    "Cu_ppm",
    "Au_ppm",
    "Mo_ppm",
    "Fe_pct",
    "dist_fault_km",
    "gravity_bouguer",
    "gravity_distance_deg",
    "has_gravity",
]

MODEL_VARIANTS = {
    "M0_no_lithology": [
        "log1p_Cu_ppm",
        "log1p_Au_ppm",
        "log1p_Mo_ppm",
        "Fe_pct",
        "log1p_dist_fault_km",
        "gravity_bouguer",
        "gravity_distance_deg",
        "has_gravity",
    ],
    "M1_intrusive_indicator": [
        "log1p_Cu_ppm",
        "log1p_Au_ppm",
        "log1p_Mo_ppm",
        "Fe_pct",
        "log1p_dist_fault_km",
        "gravity_bouguer",
        "gravity_distance_deg",
        "has_gravity",
        "lith_intrusive",
    ],
    "M2_intrusive_contact": [
        "log1p_Cu_ppm",
        "log1p_Au_ppm",
        "log1p_Mo_ppm",
        "Fe_pct",
        "log1p_dist_fault_km",
        "gravity_bouguer",
        "gravity_distance_deg",
        "has_gravity",
        "lith_intrusive",
        "log1p_dist_intrusive_contact_km",
    ],
}

BINARY_FEATURES = {"has_gravity", "lith_intrusive"}


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Poisson point-process baseline with GeMS lithology and leakage-controlled spatial validation."
    )
    parser.add_argument("--geochemistry-path", type=Path, default=defaults.geochemistry_path)
    parser.add_argument("--fault-path", type=Path, default=defaults.fault_path)
    parser.add_argument("--gravity-path", type=Path, default=defaults.gravity_path)
    parser.add_argument("--lithology-path", type=Path, default=defaults.lithology_path)
    parser.add_argument("--paper-point-dir", type=Path, default=defaults.paper_point_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--grid-km", type=float, default=defaults.grid_km)
    parser.add_argument("--block-km", type=float, default=defaults.block_km)
    parser.add_argument(
        "--buffer-radii-km",
        default=",".join(str(v) for v in defaults.buffer_radii_km),
        help="Comma-separated dead-zone radii, for example 0,25,50.",
    )
    parser.add_argument("--folds", type=int, default=defaults.n_splits)
    parser.add_argument("--l2-alpha", type=float, default=defaults.l2_alpha)
    parser.add_argument("--seed", type=int, default=defaults.random_state)
    parser.add_argument(
        "--lpo-pairs-per-positive", type=int, default=defaults.lpo_pairs_per_positive
    )
    parser.add_argument("--lpo-radius-km", type=float, default=defaults.lpo_radius_km)
    parser.add_argument("--skip-lpo", action="store_true")
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast end-to-end check using a 100 km grid, three folds and no LPO audit.",
    )
    args = parser.parse_args()
    radii = tuple(float(item.strip()) for item in args.buffer_radii_km.split(",") if item.strip())
    if not radii:
        raise ValueError("At least one validation buffer radius is required")
    cfg = Config(
        geochemistry_path=args.geochemistry_path,
        fault_path=args.fault_path,
        gravity_path=args.gravity_path,
        lithology_path=args.lithology_path,
        paper_point_dir=args.paper_point_dir,
        output_dir=args.output_dir,
        grid_km=args.grid_km,
        block_km=args.block_km,
        buffer_radii_km=radii,
        n_splits=args.folds,
        l2_alpha=args.l2_alpha,
        random_state=args.seed,
        lpo_pairs_per_positive=args.lpo_pairs_per_positive,
        lpo_radius_km=args.lpo_radius_km,
        skip_lpo=bool(args.skip_lpo),
        reuse_features=not bool(args.recompute_features),
    )
    if args.smoke:
        cfg.grid_km = 100.0
        cfg.block_km = 400.0
        cfg.buffer_radii_km = (0.0,)
        cfg.n_splits = 3
        cfg.skip_lpo = True
        cfg.output_dir = cfg.output_dir.with_name(cfg.output_dir.name + "_smoke")
    return cfg


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


def require_inputs(cfg: Config) -> None:
    required = [
        cfg.geochemistry_path,
        cfg.fault_path,
        cfg.gravity_path,
        cfg.lithology_path,
        cfg.paper_point_dir / "paper_figure2g_training_44_model_ready.csv",
        cfg.paper_point_dir / "paper_figure2g_validation_16_model_ready.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def load_deposits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(
        cfg.paper_point_dir / "paper_figure2g_training_44_model_ready.csv",
        low_memory=False,
    )
    external = pd.read_csv(
        cfg.paper_point_dir / "paper_figure2g_validation_16_model_ready.csv",
        low_memory=False,
    )
    if len(train) != 44 or len(external) != 16:
        raise ValueError(f"Expected 44/16 deposits, found {len(train)}/{len(external)}")
    required = {"DEPOSIT", "LONGITUDE", "LATITUDE", "paper_star_id"}
    for label, frame in (("training", train), ("external", external)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} deposits missing columns: {sorted(missing)}")
        frame["LONGITUDE"] = pd.to_numeric(frame["LONGITUDE"], errors="coerce")
        frame["LATITUDE"] = pd.to_numeric(frame["LATITUDE"], errors="coerce")
        if frame[["LONGITUDE", "LATITUDE"]].isna().any().any():
            raise ValueError(f"{label} deposit coordinates contain missing values")
    train = train.copy()
    external = external.copy()
    train["target_type"] = "training_deposit"
    external["target_type"] = "external_deposit"
    return train, external


def projected_bounds(cfg: Config) -> tuple[float, float, float, float]:
    forward = Transformer.from_crs("EPSG:4326", cfg.equal_area_crs, always_xy=True)
    west, east, south, north = cfg.study_bounds
    lon = np.concatenate(
        [
            np.linspace(west, east, 300),
            np.linspace(west, east, 300),
            np.full(300, west),
            np.full(300, east),
        ]
    )
    lat = np.concatenate(
        [
            np.full(300, south),
            np.full(300, north),
            np.linspace(south, north, 300),
            np.linspace(south, north, 300),
        ]
    )
    x, y = forward.transform(lon, lat)
    return float(min(x)), float(min(y)), float(max(x)), float(max(y))


def read_lithology(cfg: Config):
    import geopandas as gpd

    xmin, ymin, xmax, ymax = projected_bounds(cfg)
    margin = 100_000.0
    bbox = (xmin - margin, ymin - margin, xmax + margin, ymax + margin)
    logging.info("Reading GeMS map-unit polygons within the study window")
    lith = gpd.read_file(
        cfg.lithology_path,
        bbox=bbox,
        columns=["MapUnit", "Symbol"],
    )
    lith = lith.loc[lith.geometry.notna() & ~lith.geometry.is_empty].copy()
    lith = lith.to_crs(cfg.equal_area_crs)
    lith["Symbol"] = lith["Symbol"].fillna("Unknown").astype(str).str.strip()
    lith["lith_intrusive"] = lith["Symbol"].eq("Igneous, intrusive").astype(np.int8)
    logging.info(
        "GeMS polygons retained: %s; intrusive polygons: %s",
        len(lith),
        int(lith["lith_intrusive"].sum()),
    )
    return lith


def assign_lithology(points: pd.DataFrame, lith, cfg: Config) -> pd.DataFrame:
    import geopandas as gpd
    from shapely.geometry import Point

    out = points.copy().reset_index(drop=True)
    out["_target_id"] = np.arange(len(out), dtype=int)
    gpoints = gpd.GeoDataFrame(
        out[["_target_id"]].copy(),
        geometry=[Point(x, y) for x, y in out[["metric_x", "metric_y"]].to_numpy(float)],
        crs=cfg.equal_area_crs,
    )
    joined = gpd.sjoin(
        gpoints,
        lith[["Symbol", "lith_intrusive", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined["Symbol"] = joined["Symbol"].fillna("Unknown")
    joined["lith_intrusive"] = joined["lith_intrusive"].fillna(0).astype(np.int8)
    # Boundary overlaps are resolved in favour of an intrusive unit; otherwise
    # the first polygon is used.  This is deterministic and conservative for
    # the binary intrusive-versus-other experiment.
    joined = joined.sort_values(
        ["_target_id", "lith_intrusive"], ascending=[True, False]
    ).drop_duplicates("_target_id", keep="first")
    joined = joined.set_index("_target_id").reindex(out["_target_id"])
    out["lith_symbol"] = joined["Symbol"].to_numpy()
    out["lith_intrusive"] = joined["lith_intrusive"].to_numpy(dtype=np.int8)

    intrusive = lith.loc[lith["lith_intrusive"].eq(1), ["geometry"]].copy()
    if intrusive.empty:
        raise ValueError("No 'Igneous, intrusive' polygons were found in the study area")
    contacts = intrusive.copy()
    contacts.geometry = contacts.geometry.boundary
    contacts = contacts.explode(index_parts=False).reset_index(drop=True)
    nearest = gpd.sjoin_nearest(
        gpoints,
        contacts,
        how="left",
        distance_col="dist_intrusive_contact_m",
    )
    contact_distance = nearest.groupby("_target_id")["dist_intrusive_contact_m"].min()
    out["dist_intrusive_contact_km"] = (
        out["_target_id"].map(contact_distance).to_numpy(float) / 1000.0
    )
    return out.drop(columns="_target_id")


def build_domain_grid(cfg: Config, lith) -> pd.DataFrame:
    forward = Transformer.from_crs("EPSG:4326", cfg.equal_area_crs, always_xy=True)
    inverse = Transformer.from_crs(cfg.equal_area_crs, "EPSG:4326", always_xy=True)
    xmin, ymin, xmax, ymax = projected_bounds(cfg)
    spacing = cfg.grid_km * 1000.0
    x0 = math.floor(xmin / spacing) * spacing + spacing / 2.0
    y0 = math.floor(ymin / spacing) * spacing + spacing / 2.0
    xs = np.arange(x0, xmax + spacing, spacing)
    ys = np.arange(y0, ymax + spacing, spacing)
    xx, yy = np.meshgrid(xs, ys)
    grid = pd.DataFrame(
        {"metric_x": xx.ravel(), "metric_y": yy.ravel(), "target_type": "quadrature"}
    )
    lon, lat = inverse.transform(grid["metric_x"].to_numpy(), grid["metric_y"].to_numpy())
    grid["longitude"] = lon
    grid["latitude"] = lat
    west, east, south, north = cfg.study_bounds
    grid = grid[
        grid["longitude"].between(west, east)
        & grid["latitude"].between(south, north)
    ].reset_index(drop=True)
    grid = assign_lithology(grid, lith, cfg)
    grid = grid.loc[~grid["lith_symbol"].isin(["Water", "Unknown"])].reset_index(drop=True)
    grid["area_km2"] = cfg.grid_km**2
    grid["event_count"] = 0
    grid["cell_id"] = np.arange(len(grid), dtype=int)
    logging.info(
        "Equal-area quadrature grid: %s land cells, %.0f km2 represented",
        len(grid),
        grid["area_km2"].sum(),
    )
    return grid


def deposits_as_targets(deposits: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    forward = Transformer.from_crs("EPSG:4326", cfg.equal_area_crs, always_xy=True)
    out = deposits[["DEPOSIT", "LONGITUDE", "LATITUDE", "paper_star_id", "target_type"]].copy()
    out = out.rename(columns={"LONGITUDE": "longitude", "LATITUDE": "latitude"})
    x, y = forward.transform(out["longitude"].to_numpy(), out["latitude"].to_numpy())
    out["metric_x"] = x
    out["metric_y"] = y
    return out


def assign_event_counts(grid: pd.DataFrame, train_deposits: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = grid.copy()
    targets = deposits_as_targets(train_deposits, cfg)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        out[["metric_x", "metric_y"]].to_numpy(float)
    )
    distance, index = nn.kneighbors(targets[["metric_x", "metric_y"]].to_numpy(float))
    if float(distance.max()) > cfg.grid_km * 1000.0 * 1.5:
        raise ValueError("At least one training deposit could not be assigned to a nearby land cell")
    counts = np.bincount(index.ravel(), minlength=len(out))
    out["event_count"] = counts.astype(int)
    logging.info(
        "Assigned %s training deposits to %s event cells (maximum count in one cell: %s)",
        int(out["event_count"].sum()),
        int((out["event_count"] > 0).sum()),
        int(out["event_count"].max()),
    )
    return out


def read_geochemistry(cfg: Config) -> pd.DataFrame:
    data = pd.read_csv(cfg.geochemistry_path, low_memory=False)
    required = ["longitude", "latitude", "Cu_ppm", "Au_ppm", "Mo_ppm", "Fe_pct"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Geochemistry file missing columns: {missing}")
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    west, east, south, north = cfg.study_bounds
    data = data.dropna(subset=["longitude", "latitude"])
    return data[
        data["longitude"].between(west, east)
        & data["latitude"].between(south, north)
    ].reset_index(drop=True)


def add_geochemistry(targets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sources = read_geochemistry(cfg)
    forward = Transformer.from_crs("EPSG:4326", cfg.equal_area_crs, always_xy=True)
    sx, sy = forward.transform(sources["longitude"].to_numpy(), sources["latitude"].to_numpy())
    source_xy = np.column_stack([sx, sy])
    target_xy = targets[["metric_x", "metric_y"]].to_numpy(float)
    k = min(cfg.idw_neighbors, len(sources))
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(source_xy)
    distance, index = nn.kneighbors(target_xy)
    weights = 1.0 / np.maximum(distance, cfg.idw_floor_m) ** cfg.idw_power
    out = targets.copy()
    for feature in ["Cu_ppm", "Au_ppm", "Mo_ppm", "Fe_pct"]:
        values = sources[feature].to_numpy(float)[index]
        valid = np.isfinite(values)
        numerator = np.where(valid, values * weights, 0.0).sum(axis=1)
        denominator = np.where(valid, weights, 0.0).sum(axis=1)
        out[feature] = np.divide(
            numerator,
            denominator,
            out=np.full(len(out), np.nan),
            where=denominator > 0,
        )
    out["nearest_geochemistry_km"] = distance[:, 0] / 1000.0
    return out


def add_fault_distance(targets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    import geopandas as gpd
    from shapely.geometry import Point

    out = targets.copy()
    points = gpd.GeoDataFrame(
        {"_target_id": np.arange(len(out), dtype=int)},
        geometry=[Point(x, y) for x, y in out[["metric_x", "metric_y"]].to_numpy(float)],
        crs=cfg.equal_area_crs,
    )
    faults = gpd.read_file(cfg.fault_path)
    faults = faults.loc[faults.geometry.notna() & ~faults.geometry.is_empty, ["geometry"]]
    if faults.crs is None:
        faults = faults.set_crs("EPSG:4326", allow_override=True)
    faults = faults.to_crs(cfg.equal_area_crs).explode(index_parts=False).reset_index(drop=True)
    nearest = gpd.sjoin_nearest(points, faults, how="left", distance_col="dist_fault_m")
    distance = nearest.groupby("_target_id")["dist_fault_m"].min()
    out["dist_fault_km"] = np.arange(len(out)).astype(int)
    out["dist_fault_km"] = out["dist_fault_km"].map(distance).to_numpy(float) / 1000.0
    return out


def add_gravity(targets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = targets.copy()
    gravity = pd.read_csv(
        cfg.gravity_path,
        sep=r"\s+",
        names=["longitude", "latitude", "gravity_bouguer"],
        compression="gzip",
        dtype=np.float32,
    ).dropna()
    west, east, south, north = cfg.study_bounds
    gravity = gravity[
        gravity["longitude"].between(west - 1.0, east + 1.0)
        & gravity["latitude"].between(south - 1.0, north + 1.0)
    ].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        gravity[["longitude", "latitude"]].to_numpy(np.float32)
    )
    distance, index = nn.kneighbors(out[["longitude", "latitude"]].to_numpy(np.float32))
    distance = distance.ravel()
    value = gravity["gravity_bouguer"].to_numpy(np.float32)[index.ravel()]
    available = np.isfinite(value) & (distance <= cfg.gravity_max_distance_deg)
    value = value.astype(float)
    value[~available] = np.nan
    out["gravity_bouguer"] = value
    out["gravity_distance_deg"] = distance
    out["has_gravity"] = available.astype(np.int8)
    return out


def add_transformed_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature in ["Cu_ppm", "Au_ppm", "Mo_ppm"]:
        out[f"log1p_{feature}"] = np.log1p(np.maximum(pd.to_numeric(out[feature], errors="coerce"), 0.0))
    out["log1p_dist_fault_km"] = np.log1p(
        np.maximum(pd.to_numeric(out["dist_fault_km"], errors="coerce"), 0.0)
    )
    out["log1p_dist_intrusive_contact_km"] = np.log1p(
        np.maximum(pd.to_numeric(out["dist_intrusive_contact_km"], errors="coerce"), 0.0)
    )
    return out


def load_or_build_features(
    cfg: Config, train_deposits: pd.DataFrame, external_deposits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_cache = cfg.output_dir / f"quadrature_grid_features_{cfg.grid_km:g}km.csv"
    train_cache = cfg.output_dir / "training_deposit_features.csv"
    external_cache = cfg.output_dir / "external_deposit_features.csv"
    required = set(sum(MODEL_VARIANTS.values(), [])) | {
        "metric_x",
        "metric_y",
        "area_km2",
        "event_count",
    }
    if cfg.reuse_features and all(path.exists() for path in [grid_cache, train_cache, external_cache]):
        grid = pd.read_csv(grid_cache, low_memory=False)
        train = pd.read_csv(train_cache, low_memory=False)
        external = pd.read_csv(external_cache, low_memory=False)
        if required.issubset(grid.columns) and len(train) == 44 and len(external) == 16:
            logging.info("Reusing feature caches from %s", cfg.output_dir)
            return grid, train, external

    lith = read_lithology(cfg)
    grid = build_domain_grid(cfg, lith)
    grid = assign_event_counts(grid, train_deposits, cfg)
    train = assign_lithology(deposits_as_targets(train_deposits, cfg), lith, cfg)
    external = assign_lithology(deposits_as_targets(external_deposits, cfg), lith, cfg)
    combined = pd.concat([grid, train, external], ignore_index=True, sort=False)
    logging.info("Extracting geochemistry for %s quadrature/deposit locations", len(combined))
    combined = add_geochemistry(combined, cfg)
    logging.info("Extracting nearest-fault distance")
    combined = add_fault_distance(combined, cfg)
    logging.info("Extracting Bouguer gravity")
    combined = add_gravity(combined, cfg)
    combined = add_transformed_features(combined)
    grid = combined.loc[combined["target_type"].eq("quadrature")].copy().reset_index(drop=True)
    train = combined.loc[combined["target_type"].eq("training_deposit")].copy().reset_index(drop=True)
    external = combined.loc[combined["target_type"].eq("external_deposit")].copy().reset_index(drop=True)
    grid["event_count"] = pd.to_numeric(grid["event_count"], errors="coerce").fillna(0).astype(int)
    grid["area_km2"] = pd.to_numeric(grid["area_km2"], errors="coerce").fillna(cfg.grid_km**2)
    grid.to_csv(grid_cache, index=False, encoding="utf-8-sig")
    train.to_csv(train_cache, index=False, encoding="utf-8-sig")
    external.to_csv(external_cache, index=False, encoding="utf-8-sig")
    return grid, train, external


@dataclass
class Preprocessor:
    features: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]


@dataclass
class PPPFit:
    beta: np.ndarray
    preprocessor: Preprocessor
    log_likelihood: float
    converged: bool
    message: str


def fit_preprocessor(df: pd.DataFrame, features: list[str]) -> Preprocessor:
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        if feature in BINARY_FEATURES:
            mean, std = 0.0, 1.0
        else:
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if not np.isfinite(std) or std < 1e-12:
                std = 1.0
        medians[feature] = median
        means[feature] = mean
        stds[feature] = std
    return Preprocessor(features, medians, means, stds)


def design_matrix(df: pd.DataFrame, prep: Preprocessor) -> np.ndarray:
    columns = []
    for feature in prep.features:
        values = pd.to_numeric(df[feature], errors="coerce").fillna(prep.medians[feature]).to_numpy(float)
        values = (values - prep.means[feature]) / prep.stds[feature]
        columns.append(values)
    raw = np.column_stack(columns)
    return np.column_stack([np.ones(len(raw)), raw])


def fit_ppp(df: pd.DataFrame, features: list[str], alpha: float) -> PPPFit:
    if int(df["event_count"].sum()) < 2:
        raise ValueError("PPP training data must contain at least two deposit events")
    prep = fit_preprocessor(df, features)
    x = design_matrix(df, prep)
    y = df["event_count"].to_numpy(float)
    area = df["area_km2"].to_numpy(float)
    offset = np.log(np.maximum(area, 1e-12))
    beta0 = np.zeros(x.shape[1], dtype=float)
    beta0[0] = math.log(max(y.sum(), 1e-9) / area.sum())

    def objective(beta: np.ndarray):
        eta = x @ beta + offset
        mu = np.exp(np.clip(eta, -40.0, 30.0))
        penalty = 0.5 * alpha * float(beta[1:] @ beta[1:])
        value = float(np.sum(mu - y * eta) + penalty)
        gradient = x.T @ (mu - y)
        gradient[1:] += alpha * beta[1:]
        return value, gradient

    result = minimize(
        objective,
        beta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-7},
    )
    beta = np.asarray(result.x, dtype=float)
    eta = x @ beta + offset
    mu = np.exp(np.clip(eta, -40.0, 30.0))
    log_likelihood = float(np.sum(y * eta - mu - gammaln(y + 1.0)))
    return PPPFit(beta, prep, log_likelihood, bool(result.success), str(result.message))


def predict_intensity(fit: PPPFit, df: pd.DataFrame) -> np.ndarray:
    x = design_matrix(df, fit.preprocessor)
    return np.exp(np.clip(x @ fit.beta, -40.0, 30.0))


def coefficient_frame(model_name: str, fit: PPPFit) -> pd.DataFrame:
    names = ["intercept_log_intensity_per_km2", *fit.preprocessor.features]
    return pd.DataFrame(
        {
            "model": model_name,
            "term": names,
            "coefficient": fit.beta,
            "intensity_ratio_exp_beta": np.exp(np.clip(fit.beta, -40.0, 40.0)),
            "scale": ["raw intercept", *[
                "0-to-1" if feature in BINARY_FEATURES else "one training SD"
                for feature in fit.preprocessor.features
            ]],
        }
    )


def assign_spatial_folds(grid: pd.DataFrame, cfg: Config) -> pd.Series:
    block_m = cfg.block_km * 1000.0
    block_x = np.floor(grid["metric_x"].to_numpy(float) / block_m).astype(int)
    block_y = np.floor(grid["metric_y"].to_numpy(float) / block_m).astype(int)
    block = pd.Series([f"{x}_{y}" for x, y in zip(block_x, block_y)], index=grid.index)
    summary = pd.DataFrame(
        {
            "block": block,
            "events": grid["event_count"].to_numpy(int),
            "cells": 1,
        }
    ).groupby("block", as_index=False).sum()
    positive = summary.loc[summary["events"] > 0].copy()
    if len(positive) < cfg.n_splits:
        raise ValueError(
            f"Only {len(positive)} event-bearing blocks are available for {cfg.n_splits} folds; reduce block size or folds"
        )
    rng = np.random.default_rng(cfg.random_state)
    positive["tie"] = rng.random(len(positive))
    positive = positive.sort_values(["events", "cells", "tie"], ascending=[False, False, True])
    fold_events = np.zeros(cfg.n_splits, dtype=float)
    fold_cells = np.zeros(cfg.n_splits, dtype=float)
    assignment: dict[str, int] = {}
    for row in positive.itertuples(index=False):
        target = min(range(cfg.n_splits), key=lambda f: (fold_events[f], fold_cells[f]))
        assignment[row.block] = target + 1
        fold_events[target] += row.events
        fold_cells[target] += row.cells
    zero = summary.loc[summary["events"].eq(0)].copy()
    zero["tie"] = rng.random(len(zero))
    zero = zero.sort_values(["cells", "tie"], ascending=[False, True])
    for row in zero.itertuples(index=False):
        target = int(np.argmin(fold_cells))
        assignment[row.block] = target + 1
        fold_cells[target] += row.cells
    folds = block.map(assignment).astype(int)
    counts = grid.assign(cv_fold=folds).groupby("cv_fold").agg(
        cells=("cell_id", "size"), events=("event_count", "sum")
    )
    logging.info("Spatial folds:\n%s", counts.to_string())
    return folds


def remove_dead_zone(
    train: pd.DataFrame, test: pd.DataFrame, radius_km: float
) -> tuple[pd.DataFrame, int]:
    if radius_km <= 0:
        return train.copy(), 0
    nearest = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        test[["metric_x", "metric_y"]].to_numpy(float)
    )
    distance, _ = nearest.kneighbors(train[["metric_x", "metric_y"]].to_numpy(float))
    keep = distance.ravel() > radius_km * 1000.0
    return train.loc[keep].copy(), int((~keep).sum())


def capture_at_fraction(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(score) * fraction)))
    selected = np.argsort(score)[::-1][:count]
    denominator = max(float(y.sum()), 1.0)
    return float(y[selected].sum() / denominator)


def fold_metrics(y_count: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_binary = (y_count > 0).astype(int)
    positives = int(y_binary.sum())
    negatives = int(len(y_binary) - positives)
    auc = roc_auc_score(y_binary, score) if positives and negatives else np.nan
    ap = average_precision_score(y_binary, score) if positives else np.nan
    return {
        "auc_presence_background": float(auc),
        "pr_auc_presence_background": float(ap),
        "pair_count": float(positives * negatives),
        "positive_cells": float(positives),
        "events": float(y_count.sum()),
        "capture_top_5pct": capture_at_fraction(y_count, score, 0.05),
        "capture_top_10pct": capture_at_fraction(y_count, score, 0.10),
    }


def spatial_block_validation(grid: pd.DataFrame, cfg: Config):
    work = grid.copy()
    work["cv_fold"] = assign_spatial_folds(work, cfg)
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for radius in cfg.buffer_radii_km:
        for model_name, features in MODEL_VARIANTS.items():
            per_model_rows = []
            per_model_predictions = []
            for fold in range(1, cfg.n_splits + 1):
                test = work.loc[work["cv_fold"].eq(fold)].copy()
                initial_train = work.loc[work["cv_fold"].ne(fold)].copy()
                train, removed = remove_dead_zone(initial_train, test, radius)
                fit = fit_ppp(train, features, cfg.l2_alpha)
                score = predict_intensity(fit, test)
                metrics = fold_metrics(test["event_count"].to_numpy(int), score)
                row = {
                    "model": model_name,
                    "buffer_km": radius,
                    "fold": fold,
                    "train_cells": len(train),
                    "train_events": int(train["event_count"].sum()),
                    "test_cells": len(test),
                    "dead_zone_cells_removed": removed,
                    "log_likelihood_train": fit.log_likelihood,
                    "optimizer_converged": fit.converged,
                    **metrics,
                }
                metric_rows.append(row)
                per_model_rows.append(row)
                pred = test[["cell_id", "cv_fold", "event_count", "metric_x", "metric_y"]].copy()
                pred["model"] = model_name
                pred["buffer_km"] = radius
                pred["intensity_per_km2"] = score
                prediction_rows.append(pred)
                per_model_predictions.append(pred)
                logging.info(
                    "%s | buffer %.0f km | fold %s/%s | events %s | AUC %.3f",
                    model_name,
                    radius,
                    fold,
                    cfg.n_splits,
                    int(metrics["events"]),
                    metrics["auc_presence_background"],
                )
            fold_frame = pd.DataFrame(per_model_rows)
            pred_frame = pd.concat(per_model_predictions, ignore_index=True)
            weights = fold_frame["pair_count"].to_numpy(float)
            valid = np.isfinite(fold_frame["auc_presence_background"].to_numpy(float)) & (weights > 0)
            same_model_pairwise_auc = float(
                np.average(
                    fold_frame.loc[valid, "auc_presence_background"],
                    weights=weights[valid],
                )
            )
            pooled_auc = roc_auc_score(
                (pred_frame["event_count"].to_numpy(int) > 0).astype(int),
                pred_frame["intensity_per_km2"].to_numpy(float),
            )
            summary_rows.append(
                {
                    "model": model_name,
                    "buffer_km": radius,
                    "same_fold_pairwise_auc": same_model_pairwise_auc,
                    "pooled_auc_for_comparison": float(pooled_auc),
                    "mean_fold_pr_auc": float(fold_frame["pr_auc_presence_background"].mean()),
                    "mean_capture_top_5pct": float(fold_frame["capture_top_5pct"].mean()),
                    "mean_capture_top_10pct": float(fold_frame["capture_top_10pct"].mean()),
                    "total_test_events": int(fold_frame["events"].sum()),
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
        work,
    )


def sampled_lpo_scv(grid: pd.DataFrame, cfg: Config, features: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state + 917)
    positive_index = np.flatnonzero(grid["event_count"].to_numpy(int) > 0)
    negative_index = np.flatnonzero(grid["event_count"].to_numpy(int) == 0)
    rows: list[dict] = []
    all_xy = grid[["metric_x", "metric_y"]].to_numpy(float)
    for positive_number, pos_index in enumerate(positive_index, start=1):
        selected_negative = rng.choice(
            negative_index,
            size=min(cfg.lpo_pairs_per_positive, len(negative_index)),
            replace=False,
        )
        for neg_index in np.atleast_1d(selected_negative):
            pair_xy = all_xy[[pos_index, int(neg_index)]]
            nearest = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(pair_xy)
            distance, _ = nearest.kneighbors(all_xy)
            keep = distance.ravel() > cfg.lpo_radius_km * 1000.0
            keep[[pos_index, int(neg_index)]] = False
            train = grid.loc[keep].copy()
            if int(train["event_count"].sum()) < 2:
                continue
            fit = fit_ppp(train, features, cfg.l2_alpha)
            pair = grid.iloc[[pos_index, int(neg_index)]]
            score = predict_intensity(fit, pair)
            rows.append(
                {
                    "positive_cell_id": int(grid.iloc[pos_index]["cell_id"]),
                    "negative_cell_id": int(grid.iloc[int(neg_index)]["cell_id"]),
                    "radius_km": cfg.lpo_radius_km,
                    "positive_score": float(score[0]),
                    "negative_score": float(score[1]),
                    "pair_correct": float(score[0] > score[1]) + 0.5 * float(score[0] == score[1]),
                    "train_cells": int(keep.sum()),
                    "train_events": int(train["event_count"].sum()),
                    "optimizer_converged": fit.converged,
                }
            )
        logging.info(
            "Sampled LPO-SCV-style audit: %s/%s positive cells",
            positive_number,
            len(positive_index),
        )
    return pd.DataFrame(rows)


def external_validation(
    grid: pd.DataFrame,
    external: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deposit_rows = []
    summary_rows = []
    coefficient_rows = []
    for model_name, features in MODEL_VARIANTS.items():
        fit = fit_ppp(grid, features, cfg.l2_alpha)
        grid_score = predict_intensity(fit, grid)
        external_score = predict_intensity(fit, external)
        coefficient_rows.append(coefficient_frame(model_name, fit))
        thresholds = {
            fraction: float(np.quantile(grid_score, 1.0 - fraction))
            for fraction in (0.01, 0.05, 0.10)
        }
        for row, score in zip(external.itertuples(index=False), external_score):
            percentile = float(np.mean(grid_score <= score))
            deposit_rows.append(
                {
                    "model": model_name,
                    "DEPOSIT": row.DEPOSIT,
                    "paper_star_id": int(row.paper_star_id),
                    "longitude": float(row.longitude),
                    "latitude": float(row.latitude),
                    "lith_symbol": row.lith_symbol,
                    "lith_intrusive": int(row.lith_intrusive),
                    "dist_intrusive_contact_km": float(row.dist_intrusive_contact_km),
                    "intensity_per_km2": float(score),
                    "domain_percentile": percentile,
                    "in_top_1pct": int(score >= thresholds[0.01]),
                    "in_top_5pct": int(score >= thresholds[0.05]),
                    "in_top_10pct": int(score >= thresholds[0.10]),
                }
            )
        y = np.r_[np.ones(len(external_score)), np.zeros(len(grid_score))]
        scores = np.r_[external_score, grid_score]
        summary_rows.append(
            {
                "model": model_name,
                "external_presence_background_auc": float(roc_auc_score(y, scores)),
                "external_median_domain_percentile": float(
                    np.median([np.mean(grid_score <= value) for value in external_score])
                ),
                "external_top_1pct_hits": int(np.sum(external_score >= thresholds[0.01])),
                "external_top_5pct_hits": int(np.sum(external_score >= thresholds[0.05])),
                "external_top_10pct_hits": int(np.sum(external_score >= thresholds[0.10])),
                "fitted_total_events": float(np.sum(grid_score * grid["area_km2"].to_numpy(float))),
                "training_log_likelihood": fit.log_likelihood,
                "optimizer_converged": fit.converged,
            }
        )
        grid[f"{model_name}_intensity_per_km2"] = grid_score
    return (
        pd.DataFrame(deposit_rows),
        pd.DataFrame(summary_rows),
        pd.concat(coefficient_rows, ignore_index=True),
    )


def make_dashboard(
    cv_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "M0_no_lithology": "#4c78a8",
        "M1_intrusive_indicator": "#f58518",
        "M2_intrusive_contact": "#54a24b",
    }
    for model, group in cv_summary.groupby("model"):
        group = group.sort_values("buffer_km")
        axes[0, 0].plot(
            group["buffer_km"],
            group["same_fold_pairwise_auc"],
            marker="o",
            label=model,
            color=colors.get(model),
        )
        axes[0, 1].plot(
            group["buffer_km"],
            group["mean_capture_top_10pct"],
            marker="o",
            label=model,
            color=colors.get(model),
        )
    axes[0, 0].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0, 0].set(
        xlabel="Dead-zone radius (km)",
        ylabel="Within-fold pairwise AUC",
        title="Spatial transferability",
    )
    axes[0, 1].set(
        xlabel="Dead-zone radius (km)",
        ylabel="Mean event capture in top 10% area",
        title="Area efficiency",
    )
    ext = external_summary.set_index("model")
    x = np.arange(len(ext))
    axes[1, 0].bar(
        x,
        ext["external_top_10pct_hits"],
        color=[colors.get(name, "#777777") for name in ext.index],
    )
    axes[1, 0].set_xticks(x, [name.replace("_", "\n") for name in ext.index], rotation=0)
    axes[1, 0].set(ylabel="Deposits hit (out of 16)", title="External top 10% capture")
    coef = coefficients.loc[
        coefficients["term"].isin(["lith_intrusive", "log1p_dist_intrusive_contact_km"])
    ].copy()
    if not coef.empty:
        labels = [f"{m}\n{t}" for m, t in zip(coef["model"], coef["term"])]
        axes[1, 1].barh(labels, coef["coefficient"], color="#b279a2")
        axes[1, 1].axvline(0, color="#333333", linewidth=1)
        axes[1, 1].set(xlabel="Coefficient", title="Standardised lithology effects")
    else:
        axes[1, 1].axis("off")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("PPP lithology and spatial-validation audit", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    cfg: Config,
    grid: pd.DataFrame,
    cv_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    lpo: pd.DataFrame,
) -> None:
    lines = [
        "# PPP + 岩性 + 空间验证结果说明",
        "",
        "## 方法口径",
        "",
        "- 因变量是等面积积分网格内的训练矿床事件数，不再用 2 km 缓冲制造相关正样本。",
        "- 研究区网格是 PPP 数值积分点；零计数表示该网格对全区积分的贡献，不代表确认无矿。",
        "- 侵入岩来自 USGS GeMS `Symbol = Igneous, intrusive`，没有从地图 RGB 颜色识别。",
        "- 16 个论文验证矿床不参与特征选择、参数拟合、空间分折或阈值确定。",
        "- AUC 是 presence-background 排序 AUC，不应解释为独立真负样本条件下的分类准确率。",
        "",
        "## 数据规模",
        "",
        f"- 积分网格：{len(grid):,} 个，分辨率 {cfg.grid_km:g} km。",
        f"- 训练事件：{int(grid['event_count'].sum())} 个，事件网格 {int((grid['event_count'] > 0).sum())} 个。",
        f"- 代表面积：{grid['area_km2'].sum():,.0f} km²（中心点规则网格近似）。",
        f"- 空间块：{cfg.block_km:g} km；死区半径：{', '.join(f'{v:g}' for v in cfg.buffer_radii_km)} km。",
        "",
        "## 空间验证汇总",
        "",
        cv_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "`same_fold_pairwise_auc` 只比较同一个折模型产生的正负得分，避免把不同折模型的绝对强度直接混在一起；它仍是分块近似，不等于穷举所有正负对的完整 LPO-SCV。",
        "",
        "## 16 个封存矿床",
        "",
        external_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    if not lpo.empty:
        lines.extend(
            [
                "## 抽样 LPO-SCV 风格审计",
                "",
                f"- 半径：{cfg.lpo_radius_km:g} km。",
                f"- 正负对数：{len(lpo)}。",
                f"- 成对排序 AUC：{lpo['pair_correct'].mean():.3f}。",
                "- 每一对正背景位置由同一个重新拟合的 PPP 模型评分，并删除两点周围死区；因为背景不是确认无矿，仍按 presence-background 排序解释。",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            "- `lith_intrusive` 的系数是控制其他变量后的条件关联，不是侵入岩导致成矿的因果系数。",
            "- 1:1,000,000 岩性图只代表区域地质背景；不能声称 20 km 或更细网格具有同等制图精度。",
            "- 不同积分网格尺度必须做收敛检查；正式结论至少比较 20、10 和 5 km 的系数、强度排序与 Top 靶区稳定性。",
            "- 外部 AUC 将 16 个矿床与研究区积分网格比较，只衡量排序；Top 面积命中数是更直观的主结果。",
            "",
        ]
    )
    (cfg.output_dir / "README_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    setup_logging(cfg.output_dir)
    require_inputs(cfg)
    logging.info("Configuration:\n%s", json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str))
    train_deposits, external_deposits = load_deposits(cfg)
    grid, _, external = load_or_build_features(cfg, train_deposits, external_deposits)
    if int(grid["event_count"].sum()) != 44:
        raise RuntimeError("Training response must contain exactly the 44 training deposit events")
    fold_metrics_frame, cv_predictions, cv_summary, grid = spatial_block_validation(grid, cfg)
    external_scores, external_summary, coefficients = external_validation(grid, external, cfg)
    lpo = pd.DataFrame()
    if not cfg.skip_lpo and cfg.lpo_pairs_per_positive > 0:
        lpo = sampled_lpo_scv(grid, cfg, MODEL_VARIANTS["M2_intrusive_contact"])

    fold_metrics_frame.to_csv(cfg.output_dir / "spatial_fold_metrics.csv", index=False, encoding="utf-8-sig")
    cv_predictions.to_csv(cfg.output_dir / "spatial_cv_predictions.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(cfg.output_dir / "spatial_cv_summary.csv", index=False, encoding="utf-8-sig")
    external_scores.to_csv(cfg.output_dir / "external_16_deposit_scores.csv", index=False, encoding="utf-8-sig")
    external_summary.to_csv(cfg.output_dir / "external_validation_summary.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(cfg.output_dir / "ppp_coefficients.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(cfg.output_dir / "quadrature_predictions.csv", index=False, encoding="utf-8-sig")
    if not lpo.empty:
        lpo.to_csv(cfg.output_dir / "sampled_lpo_scv_pairs.csv", index=False, encoding="utf-8-sig")
    (cfg.output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    make_dashboard(
        cv_summary,
        external_summary,
        coefficients,
        cfg.output_dir / "validation_dashboard.png",
    )
    write_summary(cfg, grid, cv_summary, external_summary, lpo)
    logging.info("Completed PPP lithology analysis: %s", cfg.output_dir)


if __name__ == "__main__":
    main()
