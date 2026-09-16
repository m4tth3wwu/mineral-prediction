import argparse
import json
import logging
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Config:
    data_path: Path = Path(r"D:\code\ResearchPractice\data\geochemistry\沉积物地球化学数据.csv")
    fault_path: Path = Path(r"D:\code\ResearchPractice\data\faults\GeologyFaults_USCanada.shp")
    # Override the legacy mojibake path above with the actual source filename.
    data_path = Path(r"D:\code\ResearchPractice\data\geochemistry\沉积物地球化学数据.csv")
    deposit_path: Path = Path(
        r"D:\code\ResearchPractice\mineral_prediction\data\usgs_porpyhry\Porphyry_datasheet.csv"
    )
    output_dir: Path = Path(r"D:\code\ResearchPractice\mineral_prediction\output\acawlr_gravity_fault_maps")

    # Cu is an ore-controlling predictor in the paper. Deposit presence is the
    # target; Cu concentration must not be used to manufacture that target.
    base_feature_cols: tuple = ("Cu_ppm", "Au_ppm", "Mo_ppm", "Fe_pct")
    coord_candidates: tuple = (
        ("x", "y"),
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("Longitude", "Latitude"),
    )
    study_bounds: tuple = (-125.0, -102.0, 30.0, 50.0)
    deposit_buffer_km: float = 2.0
    deposit_country: str = "United States"

    neg_pos_ratio: int = 20
    max_negatives: int = 20000
    random_state: int = 42
    n_splits: int = 5

    fault_search_radius_m: float = 50_000.0
    fault_max_near_lines: int = 80
    fault_direction_fallback: float = 0.0
    reuse_fault_features: bool = True

    gravity_grid_path: Path = Path(r"D:\code\ResearchPractice\data\gravity\GRIDS\grid_xyz\bouguer.xyz.gz")
    gravity_max_distance_deg: float = 0.25
    reuse_gravity_features: bool = True

    aniso_parallel_scale: float = 3.0
    aniso_perp_scale: float = 1.0
    distance_normalize_m: float = 50_000.0
    grid_h: int = 120
    grid_w: int = 100

    batch_size: int = 32
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 6
    num_workers: int = 0
    make_maps: bool = True
    map_dpi: int = 220
    gravity_plot_bin_deg: float = 0.10
    deposit_probability_neighbors: int = 12
    deposit_probability_max_km: float = 75.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
DEVICE = torch.device(CFG.device)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ASPNN(nn.Module):
    """Anisotropic spatial proximity neural network."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return torch.relu(self.conv(x) + self.shortcut(x))


class CBAM(nn.Module):
    """Compact channel and spatial attention block."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.ca_sigmoid = nn.Sigmoid()
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = self.mlp(nn.AdaptiveAvgPool2d(1)(x))
        max_out = self.mlp(nn.AdaptiveMaxPool2d(1)(x))
        x = x * self.ca_sigmoid(avg_out + max_out)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([avg_out, max_out], dim=1))
        return x


class CAWNN(nn.Module):
    """Convolutional attention weighted neural network."""

    def __init__(self, coef_num):
        super().__init__()
        self.head = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.res1 = ResBlock(32, 8)
        self.res2 = ResBlock(8, 16)
        self.res3 = ResBlock(16, 32)
        self.cbam = CBAM(32)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, coef_num)

    def forward(self, gaspg):
        x = self.head(gaspg)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.cbam(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x)


class ACAWLR(nn.Module):
    def __init__(self, coef_num, beta_global):
        super().__init__()
        self.aspnn = ASPNN()
        self.cawnn = CAWNN(coef_num)
        self.register_buffer("beta_global", beta_global.view(-1))

    def forward(self, aniso_distances, x_features):
        b, c, h, w = aniso_distances.shape
        if c != 2:
            raise ValueError(f"aniso_distances must have 2 channels, got {c}.")

        d_reshaped = aniso_distances.permute(0, 2, 3, 1).reshape(-1, 2)
        proximity = self.aspnn(d_reshaped)
        gaspg = proximity.view(b, 1, h, w).clamp(1e-6, 1.0 - 1e-6)

        local_weight = self.cawnn(gaspg).clamp(-50.0, 50.0)
        beta_local = local_weight * self.beta_global.view(1, -1)
        return torch.sum(beta_local * x_features, dim=1, keepdim=True).clamp(-30.0, 30.0)


def smart_read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {suffix}. Use CSV or Excel.")


def infer_coord_cols(df: pd.DataFrame, cfg: Config):
    for x_col, y_col in cfg.coord_candidates:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    raise ValueError(f"No coordinate columns found. Existing columns: {list(df.columns)}")


def looks_like_lonlat(x, y) -> bool:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        np.nanmin(x) >= -180
        and np.nanmax(x) <= 180
        and np.nanmin(y) >= -90
        and np.nanmax(y) <= 90
    )


def choose_metric_crs(points_gdf):
    try:
        bounds = points_gdf.total_bounds
        lon_span = bounds[2] - bounds[0]
        lat_span = bounds[3] - bounds[1]
        if lon_span <= 12 and lat_span <= 12:
            utm = points_gdf.estimate_utm_crs()
            if utm is not None:
                return utm
        try:
            from pyproj import CRS

            return CRS.from_user_input("ESRI:102008")
        except Exception:
            return "EPSG:3857"
    except Exception:
        return "EPSG:3857"


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_deposit_points(cfg: Config) -> pd.DataFrame:
    """Load public porphyry occurrences and clip them to the paper study area."""
    require_file(cfg.deposit_path, "Deposit coordinate file")
    deposits = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            deposits = pd.read_csv(cfg.deposit_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if deposits is None:
        raise ValueError(f"Unable to decode deposit file: {cfg.deposit_path}")

    lon_candidates = ("LONGITUDE", "longitude", "lon", "x")
    lat_candidates = ("LATITUDE", "latitude", "lat", "y")
    lon_col = next((c for c in lon_candidates if c in deposits.columns), None)
    lat_col = next((c for c in lat_candidates if c in deposits.columns), None)
    if lon_col is None or lat_col is None:
        raise ValueError("Deposit file must contain longitude and latitude columns.")

    deposits[lon_col] = pd.to_numeric(deposits[lon_col], errors="coerce")
    deposits[lat_col] = pd.to_numeric(deposits[lat_col], errors="coerce")
    deposits = deposits.dropna(subset=[lon_col, lat_col]).copy()
    if cfg.deposit_country and "COUNTRY" in deposits.columns:
        deposits = deposits[deposits["COUNTRY"].eq(cfg.deposit_country)].copy()

    west, east, south, north = cfg.study_bounds
    deposits = deposits[
        deposits[lon_col].between(west, east) & deposits[lat_col].between(south, north)
    ].copy()
    deposits = deposits.rename(columns={lon_col: "deposit_longitude", lat_col: "deposit_latitude"})
    deposits = deposits.drop_duplicates(subset=["deposit_longitude", "deposit_latitude"]).reset_index(drop=True)
    if deposits.empty:
        raise ValueError("No deposits remain after clipping to the study area.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    deposits.to_csv(cfg.output_dir / "deposit_points_used.csv", index=False, encoding="utf-8-sig")
    logging.info("Deposit points used as labels: %s", len(deposits))
    return deposits


def label_by_deposit_proximity(df, x_col, y_col, deposits, buffer_km):
    """Assign labels from nearest known deposit using great-circle distance."""
    from sklearn.neighbors import BallTree

    earth_radius_km = 6371.0088
    deposit_rad = np.radians(deposits[["deposit_latitude", "deposit_longitude"]].to_numpy(float))
    sample_rad = np.radians(df[[y_col, x_col]].to_numpy(float))
    tree = BallTree(deposit_rad, metric="haversine")
    distance_rad, nearest_idx = tree.query(sample_rad, k=1)
    nearest_km = distance_rad[:, 0] * earth_radius_km
    nearest_idx = nearest_idx[:, 0]

    out = df.copy()
    out["nearest_deposit_km"] = nearest_km
    out["nearest_deposit_index"] = nearest_idx
    if "DEPOSIT" in deposits.columns:
        out["nearest_deposit_name"] = deposits.iloc[nearest_idx]["DEPOSIT"].to_numpy()
    out["Label"] = (nearest_km <= buffer_km).astype(np.int8)
    return out


def load_and_select_samples(cfg: Config):
    require_file(cfg.data_path, "Data file")
    logging.info("Reading point data: %s", cfg.data_path)
    df = smart_read_table(cfg.data_path)
    logging.info("Raw rows: %s", len(df))

    x_col, y_col = infer_coord_cols(df, cfg)
    logging.info("Using coordinate columns: %s, %s", x_col, y_col)

    feature_cols = list(cfg.base_feature_cols)
    needed = list(dict.fromkeys([x_col, y_col] + feature_cols))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[x_col, y_col, "Cu_ppm"]).reset_index(drop=True)
    if not looks_like_lonlat(df[x_col], df[y_col]):
        raise ValueError("Deposit-based labels require sample coordinates in WGS84 longitude/latitude.")

    west, east, south, north = cfg.study_bounds
    df = df[df[x_col].between(west, east) & df[y_col].between(south, north)].reset_index(drop=True)
    logging.info("Rows after coordinate/Cu cleaning and study-area clipping: %s (removed %s)", len(df), before - len(df))

    deposits = load_deposit_points(cfg)
    df = label_by_deposit_proximity(df, x_col, y_col, deposits, cfg.deposit_buffer_km)
    y = df["Label"].to_numpy(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    logging.info(
        "Proximity labels (buffer %.2f km): positives=%s, negatives=%s",
        cfg.deposit_buffer_km,
        n_pos,
        n_neg,
    )
    if n_pos < 2:
        raise ValueError("Too few positive samples; increase --deposit-buffer-km or check deposit coordinates.")

    target_neg = min(n_neg, max(n_pos * cfg.neg_pos_ratio, n_pos), cfg.max_negatives)
    rng = np.random.default_rng(cfg.random_state)
    sampled_neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
    selected_idx = np.concatenate([pos_idx, sampled_neg_idx])
    rng.shuffle(selected_idx)

    df_selected = df.iloc[selected_idx].copy().reset_index(drop=True)
    df_selected["sample_id"] = np.arange(len(df_selected), dtype=int)
    logging.info(
        "Selected rows: %s, positives: %s, negatives: %s",
        len(df_selected),
        int(df_selected["Label"].sum()),
        len(df_selected) - int(df_selected["Label"].sum()),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(
        cfg.output_dir / "selected_training_samples_before_fault.csv", index=False, encoding="utf-8-sig"
    )
    return df_selected, x_col, y_col


def axial_mean_angle(angles, weights=None, fallback=0.0):
    angles = np.asarray(angles, dtype=float)
    weights = np.ones_like(angles) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    if valid.sum() == 0:
        return fallback, 0.0

    angles = angles[valid]
    weights = weights[valid]
    c = np.sum(weights * np.cos(2.0 * angles))
    s = np.sum(weights * np.sin(2.0 * angles))
    mean_angle = 0.5 * np.arctan2(s, c)
    if mean_angle < 0:
        mean_angle += np.pi
    confidence = np.sqrt(c * c + s * s) / (np.sum(weights) + 1e-12)
    return float(mean_angle), float(confidence)


def line_orientation_and_length(line):
    from shapely.geometry import LineString, MultiLineString

    if line is None or line.is_empty:
        return [], []
    if isinstance(line, LineString):
        geoms = [line]
    elif isinstance(line, MultiLineString):
        geoms = list(line.geoms)
    else:
        geoms = [g for g in getattr(line, "geoms", []) if isinstance(g, LineString)]

    angles, lengths = [], []
    for geom in geoms:
        coords = list(geom.coords)
        for p1, p2 in zip(coords[:-1], coords[1:]):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            theta = math.atan2(dy, dx)
            if theta < 0:
                theta += math.pi
            if theta >= math.pi:
                theta -= math.pi
            angles.append(theta)
            lengths.append(length)
    return angles, lengths


def normalize_sindex_result(result):
    arr = np.asarray(result)
    if arr.size == 0:
        return []
    out = []
    for v in arr.reshape(-1):
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def compute_fault_features(df: pd.DataFrame, x_col: str, y_col: str, cfg: Config) -> pd.DataFrame:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cfg.output_dir / "selected_training_samples_with_fault.csv"
    required_fault_cols = {
        "dist_fault",
        "fault_density",
        "local_fault_angle",
        "local_fault_confidence",
        "metric_x",
        "metric_y",
    }
    if cfg.reuse_fault_features and cached_path.exists():
        cached = pd.read_csv(cached_path)
        cache_matches = False
        if required_fault_cols.issubset(cached.columns) and len(cached) == len(df):
            try:
                same_labels = cached["Label"].astype(int).equals(df["Label"].astype(int))
                same_x = np.allclose(pd.to_numeric(cached[x_col], errors="coerce"), pd.to_numeric(df[x_col], errors="coerce"))
                same_y = np.allclose(pd.to_numeric(cached[y_col], errors="coerce"), pd.to_numeric(df[y_col], errors="coerce"))
                cache_matches = same_labels and same_x and same_y
            except Exception:
                cache_matches = False
        if cache_matches:
            logging.info("Reusing cached fault features: %s", cached_path)
            return cached

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError("Install geopandas, shapely, pyproj, and openpyxl before computing fault features.") from exc

    require_file(cfg.fault_path, "Fault shapefile")
    df = df.copy().reset_index(drop=True)
    raw_x = pd.to_numeric(df[x_col], errors="coerce")
    raw_y = pd.to_numeric(df[y_col], errors="coerce")
    crs_in = "EPSG:4326" if looks_like_lonlat(raw_x, raw_y) else None

    points_gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(raw_x, raw_y)], crs=crs_in)
    if points_gdf.crs is None:
        logging.warning("Point data has no CRS. Treating x/y as planar coordinates.")
        points_metric = points_gdf
    else:
        points_metric = points_gdf.to_crs(choose_metric_crs(points_gdf))

    coords_metric = np.column_stack([points_metric.geometry.x.values, points_metric.geometry.y.values]).astype(np.float32)

    logging.info("Reading fault data: %s", cfg.fault_path)
    faults = gpd.read_file(cfg.fault_path)
    faults = faults[~faults.geometry.isna()].copy()
    faults = faults[~faults.geometry.is_empty].copy()
    if faults.empty:
        raise ValueError("Fault shapefile contains no usable geometries.")

    if points_metric.crs is None:
        faults_metric = faults.copy()
    else:
        if faults.crs is None:
            logging.warning("Fault shapefile has no CRS. Assuming it matches point source CRS.")
            faults = faults.set_crs(points_gdf.crs, allow_override=True)
        faults_metric = faults.to_crs(points_metric.crs)

    faults_metric = faults_metric.explode(index_parts=False).reset_index(drop=True)
    if faults_metric.empty:
        raise ValueError("Fault geometries are empty after explode.")

    sindex = faults_metric.sindex
    n = len(points_metric)
    dist_fault = np.zeros(n, dtype=np.float32)
    fault_density = np.zeros(n, dtype=np.float32)
    local_angle = np.zeros(n, dtype=np.float32)
    local_confidence = np.zeros(n, dtype=np.float32)
    area = math.pi * cfg.fault_search_radius_m**2

    logging.info("Computing fault features for %s selected samples.", n)
    for i, pt in enumerate(points_metric.geometry):
        # Spatial-index nearest lookup avoids testing every fault against every
        # sample (the legacy implementation was O(samples x faults)).
        try:
            nearest_pair, nearest_distance = sindex.nearest(
                pt, return_all=False, return_distance=True
            )
            nearest_pos = int(np.asarray(nearest_pair)[1, 0])
            dist_fault[i] = float(np.asarray(nearest_distance).reshape(-1)[0])
        except Exception:
            distances = faults_metric.geometry.distance(pt)
            nearest_label = distances.idxmin()
            nearest_pos = int(faults_metric.index.get_loc(nearest_label))
            dist_fault[i] = float(distances.loc[nearest_label])

        buf = pt.buffer(cfg.fault_search_radius_m)
        try:
            cand_idx = normalize_sindex_result(sindex.query(buf, predicate="intersects"))
        except Exception:
            cand_idx = []
        cand_idx = [j for j in cand_idx if 0 <= j < len(faults_metric)]
        if not cand_idx:
            cand_idx = [nearest_pos]

        cand = faults_metric.iloc[cand_idx].copy()
        if len(cand) > cfg.fault_max_near_lines:
            cand["_d"] = cand.geometry.distance(pt)
            cand = cand.nsmallest(cfg.fault_max_near_lines, "_d")

        try:
            local_len = float(cand.geometry.intersection(buf).length.sum())
        except Exception:
            local_len = float(cand.geometry.length.sum())
        fault_density[i] = local_len / (area + 1e-12)

        angles_all, lengths_all = [], []
        for geom in cand.geometry:
            angles, lengths = line_orientation_and_length(geom)
            angles_all.extend(angles)
            lengths_all.extend(lengths)
        theta, conf = axial_mean_angle(angles_all, lengths_all, cfg.fault_direction_fallback)
        local_angle[i] = theta
        local_confidence[i] = conf

        if (i + 1) % 1000 == 0:
            logging.info("Computed fault features for %s/%s samples.", i + 1, n)

    df["dist_fault"] = dist_fault
    df["fault_density"] = fault_density
    df["local_fault_angle"] = local_angle
    df["local_fault_confidence"] = local_confidence
    df["metric_x"] = coords_metric[:, 0]
    df["metric_y"] = coords_metric[:, 1]
    df.to_csv(cached_path, index=False, encoding="utf-8-sig")
    return df


def cache_matches_points(cached: pd.DataFrame, df: pd.DataFrame, x_col: str, y_col: str) -> bool:
    if len(cached) != len(df):
        return False
    try:
        same_labels = cached["Label"].astype(int).equals(df["Label"].astype(int))
        same_x = np.allclose(pd.to_numeric(cached[x_col], errors="coerce"), pd.to_numeric(df[x_col], errors="coerce"))
        same_y = np.allclose(pd.to_numeric(cached[y_col], errors="coerce"), pd.to_numeric(df[y_col], errors="coerce"))
        return bool(same_labels and same_x and same_y)
    except Exception:
        return False


def compute_gravity_features(df: pd.DataFrame, x_col: str, y_col: str, cfg: Config) -> pd.DataFrame:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cfg.output_dir / "selected_training_samples_with_gravity.csv"
    required_cols = {"gravity_bouguer", "gravity_distance_deg", "has_gravity"}
    if cfg.reuse_gravity_features and cached_path.exists():
        cached = pd.read_csv(cached_path)
        if required_cols.issubset(cached.columns) and cache_matches_points(cached, df, x_col, y_col):
            logging.info("Reusing cached gravity features: %s", cached_path)
            return cached

    require_file(cfg.gravity_grid_path, "Gravity grid")
    df = df.copy().reset_index(drop=True)
    raw_x = pd.to_numeric(df[x_col], errors="coerce")
    raw_y = pd.to_numeric(df[y_col], errors="coerce")
    if not looks_like_lonlat(raw_x, raw_y):
        logging.warning(
            "Point coordinates do not look like lon/lat. Gravity features are set to missing because the gravity grid is lon/lat."
        )
        df["gravity_bouguer"] = np.nan
        df["gravity_distance_deg"] = np.nan
        df["has_gravity"] = 0.0
        df.to_csv(cached_path, index=False, encoding="utf-8-sig")
        return df

    logging.info("Reading gravity grid: %s", cfg.gravity_grid_path)
    gravity = pd.read_csv(
        cfg.gravity_grid_path,
        sep=r"\s+",
        names=["gravity_lon", "gravity_lat", "gravity_bouguer"],
        compression="gzip",
        dtype=np.float32,
    ).dropna()
    logging.info("Gravity grid points: %s", len(gravity))

    from sklearn.neighbors import NearestNeighbors

    grid_xy = gravity[["gravity_lon", "gravity_lat"]].to_numpy(dtype=np.float32)
    point_xy = np.column_stack([raw_x.to_numpy(dtype=np.float32), raw_y.to_numpy(dtype=np.float32)])

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(grid_xy)
    distances, indices = nn.kneighbors(point_xy, return_distance=True)
    distances = distances.reshape(-1)
    indices = indices.reshape(-1)

    values = gravity["gravity_bouguer"].to_numpy(dtype=np.float32)[indices]
    has_gravity = np.isfinite(values) & np.isfinite(distances) & (distances <= cfg.gravity_max_distance_deg)
    values = values.astype(np.float32)
    values[~has_gravity] = np.nan

    df["gravity_bouguer"] = values
    df["gravity_distance_deg"] = distances.astype(np.float32)
    df["has_gravity"] = has_gravity.astype(np.float32)
    logging.info(
        "Matched gravity values for %s/%s samples within %.3f degrees.",
        int(has_gravity.sum()),
        len(df),
        cfg.gravity_max_distance_deg,
    )
    df.to_csv(cached_path, index=False, encoding="utf-8-sig")
    return df


def prepare_features_for_fold(df, train_idx, val_idx, feature_cols):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    medians = train_df[feature_cols].median(numeric_only=True)
    train_x = train_df[feature_cols].fillna(medians)
    val_x = val_df[feature_cols].fillna(medians)
    means = train_x.mean()
    stds = train_x.std().replace(0, 1.0)

    train_x = (train_x - means) / stds
    val_x = (val_x - means) / stds

    X_train_raw = torch.tensor(train_x.values, dtype=torch.float32)
    X_val_raw = torch.tensor(val_x.values, dtype=torch.float32)
    X_train = torch.cat([X_train_raw, torch.ones(X_train_raw.size(0), 1)], dim=1)
    X_val = torch.cat([X_val_raw, torch.ones(X_val_raw.size(0), 1)], dim=1)

    y_train = torch.tensor(train_df["Label"].values, dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor(val_df["Label"].values, dtype=torch.float32).view(-1, 1)
    coords_train = torch.tensor(train_df[["metric_x", "metric_y"]].values, dtype=torch.float32)
    coords_val = torch.tensor(val_df[["metric_x", "metric_y"]].values, dtype=torch.float32)
    theta_train = torch.tensor(train_df["local_fault_angle"].values, dtype=torch.float32).view(-1, 1)
    theta_val = torch.tensor(val_df["local_fault_angle"].values, dtype=torch.float32).view(-1, 1)

    prep = {
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "feature_cols": feature_cols,
    }
    return X_train, y_train, coords_train, theta_train, X_val, y_val, coords_val, theta_val, prep


def compute_global_beta_logistic(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_np = X[:, :-1].detach().cpu().numpy()
    y_np = y.view(-1).detach().cpu().numpy().astype(int)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=CFG.random_state,
    )
    clf.fit(x_np, y_np)
    beta = np.concatenate([clf.coef_.reshape(-1), clf.intercept_.reshape(-1)])
    return torch.tensor(beta, dtype=torch.float32)


def make_global_grid(coords_train, h, w, padding_ratio=0.05, device=DEVICE):
    xmin = coords_train[:, 0].min().item()
    xmax = coords_train[:, 0].max().item()
    ymin = coords_train[:, 1].min().item()
    ymax = coords_train[:, 1].max().item()
    xpad = (xmax - xmin) * padding_ratio + 1e-6
    ypad = (ymax - ymin) * padding_ratio + 1e-6
    xs = torch.linspace(xmin - xpad, xmax + xpad, steps=h, device=device)
    ys = torch.linspace(ymin - ypad, ymax + ypad, steps=w, device=device)
    return xs.view(h, 1).expand(h, w), ys.view(1, w).expand(h, w)


def generate_anisotropic_grid(grid_x, grid_y, target_xy, theta, cfg: Config):
    target_x = target_xy[:, 0].view(-1, 1, 1)
    target_y = target_xy[:, 1].view(-1, 1, 1)
    theta = theta.view(-1, 1, 1)

    du = grid_x.unsqueeze(0) - target_x
    dv = grid_y.unsqueeze(0) - target_y
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    d_parallel = (du * cos_t + dv * sin_t) / (cfg.aniso_parallel_scale * cfg.distance_normalize_m)
    d_perp = (-du * sin_t + dv * cos_t) / (cfg.aniso_perp_scale * cfg.distance_normalize_m)
    return torch.stack([d_parallel, d_perp], dim=1).clamp(-20.0, 20.0)


@torch.no_grad()
def predict_prob(model, grid_x, grid_y, coords, theta, X, cfg: Config, y=None):
    model.eval()
    ds = TensorDataset(coords, theta, X) if y is None else TensorDataset(coords, theta, X, y)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    probs, labels = [], []

    for batch in loader:
        if y is None:
            batch_coords, batch_theta, batch_x = batch
            batch_y = None
        else:
            batch_coords, batch_theta, batch_x, batch_y = batch

        aniso = generate_anisotropic_grid(grid_x, grid_y, batch_coords, batch_theta, cfg)
        prob = torch.sigmoid(model(aniso, batch_x)).view(-1)
        probs.append(prob.detach().cpu().numpy())
        if batch_y is not None:
            labels.append(batch_y.view(-1).detach().cpu().numpy())

    probs = np.concatenate(probs)
    if y is None:
        return probs, None
    return probs, np.concatenate(labels)


def safe_roc_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def safe_average_precision(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, y_prob)
    except Exception:
        return np.nan


def choose_threshold_from_train(y_true, y_prob) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": safe_roc_auc(y_true, y_prob),
        "ap_pr_auc": safe_average_precision(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "n_val": int(len(y_true)),
        "n_val_pos": int(y_true.sum()),
        "n_val_neg": int(len(y_true) - y_true.sum()),
    }


def train_one_fold(fold, data, cfg: Config):
    X_train, y_train, coords_train, theta_train, X_val, y_val, coords_val, theta_val, prep = data
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    coords_train = coords_train.to(DEVICE)
    theta_train = theta_train.to(DEVICE)
    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)
    coords_val = coords_val.to(DEVICE)
    theta_val = theta_val.to(DEVICE)

    beta_global = compute_global_beta_logistic(X_train, y_train).to(DEVICE)
    model = ACAWLR(coef_num=X_train.size(1), beta_global=beta_global).to(DEVICE)

    n_pos = float(y_train.sum().item())
    n_neg = float(y_train.numel() - y_train.sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_ds = TensorDataset(coords_train, theta_train, X_train, y_train)
    generator = torch.Generator()
    generator.manual_seed(cfg.random_state + fold)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        generator=generator,
    )
    grid_x, grid_y = make_global_grid(coords_train, cfg.grid_h, cfg.grid_w, device=DEVICE)

    best_val_ap = -1.0
    best_state = None
    patience_counter = 0
    logging.info("Fold %s started. Train pos=%s, neg=%s, pos_weight=%.3f", fold, int(n_pos), int(n_neg), pos_weight.item())

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for batch_coords, batch_theta, batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            aniso = generate_anisotropic_grid(grid_x, grid_y, batch_coords, batch_theta, cfg)
            logits = model(aniso, batch_x)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                logging.warning("Fold %s epoch %s skipped a non-finite batch loss.", fold, epoch)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        avg_train_loss = total_loss / max(total_n, 1)
        val_prob, val_label = predict_prob(model, grid_x, grid_y, coords_val, theta_val, X_val, cfg, y_val)
        val_ap = safe_average_precision(val_label, val_prob)
        val_auc = safe_roc_auc(val_label, val_prob)
        logging.info(
            "Fold %s | Epoch %02d/%s | train_loss=%.6f | val_AP=%.4f | val_AUC=%.4f",
            fold,
            epoch,
            cfg.epochs,
            avg_train_loss,
            val_ap,
            val_auc,
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg.early_stop_patience:
            logging.info("Fold %s early stopped at epoch %s.", fold, epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_prob, train_label = predict_prob(model, grid_x, grid_y, coords_train, theta_train, X_train, cfg, y_train)
    threshold = choose_threshold_from_train(train_label, train_prob)
    val_prob, val_label = predict_prob(model, grid_x, grid_y, coords_val, theta_val, X_val, cfg, y_val)
    metrics = compute_metrics(val_label, val_prob, threshold)
    metrics["fold"] = fold
    metrics["best_val_ap"] = best_val_ap

    model_path = cfg.output_dir / f"model_fold_{fold}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "beta_global": beta_global.detach().cpu(),
            "fold": fold,
            "threshold": threshold,
            "preprocess": prep,
        },
        model_path,
    )
    return metrics, val_prob, val_label


def run_fivefold_validation(df: pd.DataFrame, cfg: Config):
    feature_cols = list(cfg.base_feature_cols) + [
        "dist_fault",
        "fault_density",
        "local_fault_confidence",
        "gravity_bouguer",
        "gravity_distance_deg",
        "has_gravity",
    ]

    missing = [col for col in feature_cols + ["Label"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing modeling columns: {missing}")

    y = df["Label"].values.astype(int)
    n_pos = int(y.sum())
    n_splits = min(cfg.n_splits, n_pos)
    if n_splits < 2:
        raise ValueError("Too few positive samples for cross-validation.")
    if n_splits < cfg.n_splits:
        logging.warning("Reducing folds from %s to %s because positives are limited.", cfg.n_splits, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    all_metrics, cv_pred_rows = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        data = prepare_features_for_fold(df, train_idx, val_idx, feature_cols)
        metrics, val_prob, val_label = train_one_fold(fold, data, cfg)
        all_metrics.append(metrics)

        val_df = df.iloc[val_idx].copy().reset_index(drop=True)
        val_df["fold"] = fold
        val_df["cv_prob"] = val_prob
        val_df["cv_label"] = val_label.astype(int)
        val_df["fold_threshold"] = metrics["threshold"]
        cv_pred_rows.append(val_df)

    metrics_df = pd.DataFrame(all_metrics)
    cv_pred_df = pd.concat(cv_pred_rows, ignore_index=True)
    metrics_path = cfg.output_dir / "cv_metrics.csv"
    pred_path = cfg.output_dir / "cv_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    cv_pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    logging.info("Five-fold metrics:\n%s", metrics_df.to_string(index=False))
    logging.info("Mean metrics:\n%s", metrics_df.select_dtypes(include=[np.number]).mean().to_string())
    return metrics_df, cv_pred_df


def load_map_layers(cfg: Config):
    """Load and clip Bouguer gravity and faults for reproducible static maps."""
    import geopandas as gpd

    west, east, south, north = cfg.study_bounds
    gravity = pd.read_csv(
        cfg.gravity_grid_path,
        sep=r"\s+",
        names=["longitude", "latitude", "gravity_bouguer"],
        compression="gzip",
        dtype=np.float32,
    )
    gravity = gravity[
        gravity["longitude"].between(west, east)
        & gravity["latitude"].between(south, north)
    ].copy()
    step = cfg.gravity_plot_bin_deg
    gravity["lon_bin"] = np.floor((gravity["longitude"] - west) / step) * step + west
    gravity["lat_bin"] = np.floor((gravity["latitude"] - south) / step) * step + south
    gravity_grid = gravity.groupby(["lat_bin", "lon_bin"], observed=True)["gravity_bouguer"].mean().unstack()
    gravity_grid = gravity_grid.sort_index().sort_index(axis=1)

    faults = gpd.read_file(cfg.fault_path)
    if faults.crs is None:
        faults = faults.set_crs("EPSG:4326", allow_override=True)
    else:
        faults = faults.to_crs("EPSG:4326")
    faults = faults.cx[west:east, south:north].copy()
    return gravity_grid, faults


def infer_plot_coord_cols(df: pd.DataFrame):
    for x_col, y_col in (("longitude", "latitude"), ("x", "y"), ("lon", "lat")):
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    raise ValueError("No longitude/latitude columns found for mapping.")


def draw_geologic_base(ax, gravity_grid, faults, cfg: Config, gravity_alpha=0.82):
    west, east, south, north = cfg.study_bounds
    image = ax.imshow(
        gravity_grid.to_numpy(),
        origin="lower",
        extent=[
            float(gravity_grid.columns.min()),
            float(gravity_grid.columns.max() + cfg.gravity_plot_bin_deg),
            float(gravity_grid.index.min()),
            float(gravity_grid.index.max() + cfg.gravity_plot_bin_deg),
        ],
        cmap="RdBu_r",
        alpha=gravity_alpha,
        aspect="auto",
        zorder=0,
    )
    if len(faults):
        faults.plot(ax=ax, color="#262626", linewidth=0.28, alpha=0.45, zorder=2)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(color="white", linewidth=0.35, alpha=0.35)
    return image


def estimate_deposit_probabilities(cv_predictions, deposits, cfg: Config):
    """Estimate each deposit probability from nearby out-of-fold predictions."""
    from sklearn.neighbors import BallTree

    x_col, y_col = infer_plot_coord_cols(cv_predictions)
    samples_rad = np.radians(cv_predictions[[y_col, x_col]].to_numpy(float))
    deposits_rad = np.radians(deposits[["deposit_latitude", "deposit_longitude"]].to_numpy(float))
    k = min(cfg.deposit_probability_neighbors, len(cv_predictions))
    tree = BallTree(samples_rad, metric="haversine")
    distance_rad, indices = tree.query(deposits_rad, k=k)
    distance_km = distance_rad * 6371.0088
    neighbor_prob = cv_predictions["cv_prob"].to_numpy(float)[indices]
    valid = distance_km <= cfg.deposit_probability_max_km
    weights = np.where(valid, 1.0 / np.maximum(distance_km, 0.25) ** 2, 0.0)
    weight_sum = weights.sum(axis=1)
    probability = np.divide(
        (weights * neighbor_prob).sum(axis=1),
        weight_sum,
        out=np.full(len(deposits), np.nan),
        where=weight_sum > 0,
    )
    extrapolated = weight_sum == 0
    # Always provide a probability, while explicitly flagging deposits whose
    # nearest usable out-of-fold sample lies beyond the preferred radius.
    probability[extrapolated] = neighbor_prob[extrapolated, 0]

    result = deposits.copy()
    result["predicted_probability"] = probability
    result["nearest_oof_sample_km"] = distance_km[:, 0]
    result["neighbors_used"] = np.where(extrapolated, 1, valid.sum(axis=1))
    result["probability_method"] = np.where(
        extrapolated,
        "nearest_out_of_fold_sample_fallback",
        "inverse_distance_weighted_out_of_fold_samples",
    )
    result["is_distance_extrapolation"] = extrapolated
    result["probability_is_independent_validation"] = False
    result.to_csv(cfg.output_dir / "deposit_prediction_probabilities.csv", index=False, encoding="utf-8-sig")
    return result


def plot_input_layers(samples, deposits, gravity_grid, faults, cfg: Config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_col, y_col = infer_plot_coord_cols(samples)
    fig, ax = plt.subplots(figsize=(11.5, 8.5), constrained_layout=True)
    image = draw_geologic_base(ax, gravity_grid, faults, cfg)
    neg = samples[samples["Label"].eq(0)]
    pos = samples[samples["Label"].eq(1)]
    ax.scatter(neg[x_col], neg[y_col], s=2.2, c="#b9b9b9", alpha=0.22, linewidths=0, label="Sampled negatives", zorder=3)
    ax.scatter(pos[x_col], pos[y_col], s=8, c="#ffd166", edgecolors="#5f3700", linewidths=0.25, alpha=0.75, label="Positive buffer samples", zorder=4)
    ax.scatter(
        deposits["deposit_longitude"],
        deposits["deposit_latitude"],
        s=42,
        marker="*",
        c="#e63946",
        edgecolors="white",
        linewidths=0.45,
        label="USGS porphyry occurrences",
        zorder=5,
    )
    ax.set_title("Model inputs: Bouguer gravity, faults, sediment samples, and deposits")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Bouguer gravity anomaly (mGal)")
    out = cfg.output_dir / "map_model_inputs.png"
    fig.savefig(out, dpi=cfg.map_dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_validation_outputs(metrics, cv_predictions, deposit_probabilities, gravity_grid, faults, cfg: Config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    x_col, y_col = infer_plot_coord_cols(cv_predictions)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    image = draw_geologic_base(axes[0, 0], gravity_grid, faults, cfg, gravity_alpha=0.9)
    deposit_scatter = axes[0, 0].scatter(
        deposit_probabilities["deposit_longitude"],
        deposit_probabilities["deposit_latitude"],
        c=deposit_probabilities["predicted_probability"],
        cmap="magma",
        vmin=0,
        vmax=1,
        s=54,
        marker="*",
        edgecolors="white",
        linewidths=0.35,
        zorder=4,
    )
    axes[0, 0].set_title("A. Deposit probabilities over gravity and faults")
    fig.colorbar(image, ax=axes[0, 0], shrink=0.73, label="Bouguer gravity (mGal)")
    fig.colorbar(
        deposit_scatter,
        ax=axes[0, 0],
        orientation="horizontal",
        shrink=0.62,
        pad=0.12,
        label="Deposit probability",
    )

    prob_map = axes[0, 1].hexbin(
        cv_predictions[x_col],
        cv_predictions[y_col],
        C=cv_predictions["cv_prob"],
        reduce_C_function=np.mean,
        gridsize=80,
        extent=cfg.study_bounds,
        mincnt=1,
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    if len(faults):
        faults.plot(ax=axes[0, 1], color="#343434", linewidth=0.22, alpha=0.28)
    axes[0, 1].scatter(
        deposit_probabilities["deposit_longitude"],
        deposit_probabilities["deposit_latitude"],
        marker="*",
        s=23,
        facecolors="none",
        edgecolors="#00e5ff",
        linewidths=0.6,
    )
    axes[0, 1].set_xlim(cfg.study_bounds[0], cfg.study_bounds[1])
    axes[0, 1].set_ylim(cfg.study_bounds[2], cfg.study_bounds[3])
    axes[0, 1].set_xlabel("Longitude (°)")
    axes[0, 1].set_ylabel("Latitude (°)")
    axes[0, 1].set_title("B. Mean out-of-fold probability by hexagon")
    fig.colorbar(prob_map, ax=axes[0, 1], shrink=0.73, label="Predicted probability")

    y_true = cv_predictions["cv_label"].to_numpy(int)
    y_prob = cv_predictions["cv_prob"].to_numpy(float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc = safe_roc_auc(y_true, y_prob)
    ap = safe_average_precision(y_true, y_prob)
    axes[1, 0].plot(fpr, tpr, color="#1d3557", linewidth=2, label=f"OOF AUC = {auc:.3f}")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="#9a9a9a", linewidth=1)
    axes[1, 0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="C. Out-of-fold ROC")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(recall, precision, color="#e76f51", linewidth=2, label=f"OOF AP = {ap:.3f}")
    axes[1, 1].axhline(y_true.mean(), linestyle="--", color="#9a9a9a", linewidth=1, label="Positive prevalence")
    axes[1, 1].set(xlabel="Recall", ylabel="Precision", title="D. Out-of-fold precision-recall")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("ACAWLR validation and deposit-probability visualization", fontsize=15, fontweight="bold")
    out = cfg.output_dir / "map_and_validation_dashboard.png"
    fig.savefig(out, dpi=cfg.map_dpi, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "oof_auc": auc,
        "oof_average_precision": ap,
        "mean_fold_auc": float(metrics["auc"].mean()),
        "mean_fold_recall": float(metrics["recall"].mean()),
        "mean_fold_accuracy": float(metrics["accuracy"].mean()),
    }
    (cfg.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def make_all_visualizations(samples, metrics, cv_predictions, cfg: Config):
    deposits = load_deposit_points(cfg)
    gravity_grid, faults = load_map_layers(cfg)
    plot_input_layers(samples, deposits, gravity_grid, faults, cfg)
    deposit_probabilities = estimate_deposit_probabilities(cv_predictions, deposits, cfg)
    plot_validation_outputs(metrics, cv_predictions, deposit_probabilities, gravity_grid, faults, cfg)
    return deposit_probabilities


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-aligned ACAWLR with Cu, Au, Mo, Fe, Bouguer gravity, faults, and map outputs."
    )
    parser.add_argument("--data-path", type=Path, default=CFG.data_path)
    parser.add_argument("--deposit-path", type=Path, default=CFG.deposit_path)
    parser.add_argument("--fault-path", type=Path, default=CFG.fault_path)
    parser.add_argument("--output-dir", type=Path, default=CFG.output_dir)
    parser.add_argument("--deposit-buffer-km", type=float, default=CFG.deposit_buffer_km)
    parser.add_argument("--neg-pos-ratio", type=int, default=CFG.neg_pos_ratio)
    parser.add_argument("--max-negatives", type=int, default=CFG.max_negatives)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--grid-h", type=int, default=CFG.grid_h)
    parser.add_argument("--grid-w", type=int, default=CFG.grid_w)
    parser.add_argument("--gravity-grid-path", type=Path, default=CFG.gravity_grid_path)
    parser.add_argument("--gravity-max-distance-deg", type=float, default=CFG.gravity_max_distance_deg)
    parser.add_argument("--no-reuse-fault-features", action="store_true")
    parser.add_argument("--no-reuse-gravity-features", action="store_true")
    parser.add_argument("--no-maps", action="store_true", help="Skip PNG map and validation dashboard output.")
    return parser.parse_args()


def main():
    global CFG, DEVICE
    args = parse_args()
    CFG = Config(
        data_path=args.data_path,
        deposit_path=args.deposit_path,
        fault_path=args.fault_path,
        output_dir=args.output_dir,
        deposit_buffer_km=args.deposit_buffer_km,
        neg_pos_ratio=args.neg_pos_ratio,
        max_negatives=args.max_negatives,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        gravity_grid_path=args.gravity_grid_path,
        gravity_max_distance_deg=args.gravity_max_distance_deg,
        reuse_fault_features=not args.no_reuse_fault_features,
        reuse_gravity_features=not args.no_reuse_gravity_features,
        make_maps=not args.no_maps,
    )
    DEVICE = torch.device(CFG.device)

    setup_logging(CFG.output_dir)
    seed_everything(CFG.random_state)
    logging.info("Device: %s", DEVICE)
    logging.info("Config:\n%s", json.dumps({k: str(v) for k, v in asdict(CFG).items()}, ensure_ascii=False, indent=2))

    df_selected, x_col, y_col = load_and_select_samples(CFG)
    df_selected = compute_fault_features(df_selected, x_col, y_col, CFG)
    df_selected = compute_gravity_features(df_selected, x_col, y_col, CFG)
    final_selected_path = CFG.output_dir / "selected_training_samples_final.csv"
    df_selected.to_csv(final_selected_path, index=False, encoding="utf-8-sig")

    metrics_df, cv_pred_df = run_fivefold_validation(df_selected, CFG)
    if CFG.make_maps:
        make_all_visualizations(df_selected, metrics_df, cv_pred_df, CFG)
    logging.info("Done.")


if __name__ == "__main__":
    main()
