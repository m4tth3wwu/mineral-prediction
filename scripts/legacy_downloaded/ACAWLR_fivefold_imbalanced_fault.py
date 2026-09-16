# -*- coding: utf-8 -*-
"""
ACAWLR_fivefold_imbalanced_fault.py
====================================

适用场景：
- 地球化学点数据很多，例如 60 万行；
- 正样本很少，例如只有几百个高 Cu 点；
- 有美国/加拿大断裂带 shp 文件；
- 需要先筛选训练样本，再做 five-fold validation。

核心思路：
1. 读取全部点数据，但不把 60 万个点全部拿来训练；
2. 用 Cu 阈值构造正负样本；
3. 保留全部正样本，只从负样本中抽取一定比例作为训练负样本；
4. 对筛选后的样本计算断裂带特征；
5. 做 5 折分层交叉验证 StratifiedKFold；
6. 每一折重新标准化、重新训练、重新评估；
7. 输出 cv_metrics.csv、cv_predictions.csv、selected_training_samples.csv。

依赖安装：
D:\anaconda\python.exe -m pip install pandas numpy torch scikit-learn geopandas shapely pyproj openpyxl

推荐运行：
cd D:\code\ResearchPractice\scripts\legacy_downloaded
D:\anaconda\python.exe ACAWLR_fivefold_imbalanced_fault.py
"""

import os
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 0. 配置区：你主要改这里
# ============================================================

DATA_PATH = r"D:\code\ResearchPractice\data\geochemistry\沉积物地球化学数据.csv"
FAULT_PATH = r"D:\code\ResearchPractice\data\faults\GeologyFaults_USCanada.shp"

OUTPUT_DIR = Path(r"D:\code\ResearchPractice\outputs\legacy_downloaded_ACAWLR_fivefold_imbalanced_fault")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 表格字段名
CU_COL = "Cu_ppm"
BASE_FEATURE_COLS = ["Au_ppm", "Mo_ppm", "Fe_pct"]
COORD_COLS_CANDIDATES = [
    ("x", "y"),
    ("longitude", "latitude"),
    ("lon", "lat"),
    ("Longitude", "Latitude"),
]

# 正样本阈值：Cu >= 3000 认为是正样本
CU_THRESHOLD = 3000.0

# 负样本筛选参数
# 保留全部正样本，然后从负样本中抽 NEG_POS_RATIO 倍负样本。
# 例如正样本 300 个，NEG_POS_RATIO=20，则抽取 6000 个负样本。
NEG_POS_RATIO = 20
MAX_NEGATIVES = 20000
RANDOM_STATE = 42

# 5 折交叉验证
N_SPLITS = 5

# 断层特征参数
FAULT_SEARCH_RADIUS_M = 50_000.0
FAULT_MAX_NEAR_LINES = 80
FAULT_DIRECTION_FALLBACK = 0.0

# 各向异性距离参数
ANISO_PARALLEL_SCALE = 3.0
ANISO_PERP_SCALE = 1.0
DISTANCE_NORMALIZE_M = 50_000.0

# GASPG 网格大小。数据大时不要设太大。
GRID_H = 48
GRID_W = 48

# 训练参数
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 固定随机种子
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_STATE)


# ============================================================
# 2. 模型结构：ASPNN + CAWNN + 局部加权逻辑回归
# ============================================================

class ASPNN(nn.Module):
    """Anisotropic Spatial Proximity Neural Network"""

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
    """简化 CBAM：通道注意力 + 空间注意力"""

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
    """Convolutional Attention Weighted Neural Network"""

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
    """
    ACAWLR 简化实现：
    1. ASPNN 把各向异性距离转为空间邻近权重；
    2. CAWNN 根据 GASPG 输出局部系数权重；
    3. 局部权重乘以全局 beta，得到局部回归系数；
    4. 输出二分类 logit。
    """

    def __init__(self, coef_num, beta_global):
        super().__init__()
        self.aspnn = ASPNN()
        self.cawnn = CAWNN(coef_num)
        self.register_buffer("beta_global", beta_global.view(-1))

    def forward(self, aniso_distances, x_features):
        b, c, h, w = aniso_distances.shape
        if c != 2:
            raise ValueError(f"aniso_distances 的通道数必须为 2，现在是 {c}")

        d_reshaped = aniso_distances.permute(0, 2, 3, 1).reshape(-1, 2)
        proximity = self.aspnn(d_reshaped)
        gaspg = proximity.view(b, 1, h, w)
        gaspg = torch.clamp(gaspg, min=1e-6, max=1.0 - 1e-6)

        local_weight = self.cawnn(gaspg)
        local_weight = torch.clamp(local_weight, min=-50.0, max=50.0)

        beta_local = local_weight * self.beta_global.view(1, -1)
        logit = torch.sum(beta_local * x_features, dim=1, keepdim=True)
        logit = torch.clamp(logit, min=-30.0, max=30.0)
        return logit


# ============================================================
# 3. 数据读取与基础处理
# ============================================================

def smart_read_table(path):
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if suffix == ".csv":
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)

    raise ValueError(f"不支持的数据格式：{suffix}，请使用 CSV 或 Excel。")


def infer_coord_cols(df):
    for x_col, y_col in COORD_COLS_CANDIDATES:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    raise ValueError(
        f"找不到坐标列。请确认存在这些组合之一：{COORD_COLS_CANDIDATES}\n"
        f"当前字段：{list(df.columns)}"
    )


def looks_like_lonlat(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        np.nanmin(x) >= -180 and np.nanmax(x) <= 180
        and np.nanmin(y) >= -90 and np.nanmax(y) <= 90
    )


def choose_metric_crs(points_gdf):
    """
    小区域优先 UTM，大区域北美优先 ESRI:102008。
    """
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


# ============================================================
# 4. 正负样本筛选
# ============================================================

def load_and_select_samples():
    print("读取点数据...")
    df = smart_read_table(DATA_PATH)
    print(f"原始数据量：{len(df)} 行")
    print(f"字段：{list(df.columns)}")

    x_col, y_col = infer_coord_cols(df)
    print(f"使用坐标列：x = {x_col}, y = {y_col}")

    needed = [CU_COL, x_col, y_col] + BASE_FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{missing}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[CU_COL, x_col, y_col]).reset_index(drop=True)
    print(f"删除 Cu 或坐标缺失后：{len(df)} 行，删除 {before - len(df)} 行")

    # 特征缺失先不删除，后面会用训练集 median 填补。
    y = (df[CU_COL] >= CU_THRESHOLD).astype(int).values
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    print(f"正样本数：{n_pos}；负样本数：{n_neg}")

    if n_pos < 5:
        raise ValueError("正样本少于 5 个，不适合做 five-fold validation。请降低阈值或检查数据。")

    target_neg = min(n_neg, max(n_pos * NEG_POS_RATIO, n_pos), MAX_NEGATIVES)
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)

    selected_idx = np.concatenate([pos_idx, sampled_neg_idx])
    rng.shuffle(selected_idx)

    df_selected = df.iloc[selected_idx].copy().reset_index(drop=True)
    df_selected["Label"] = (df_selected[CU_COL] >= CU_THRESHOLD).astype(int)

    print(
        f"筛选后训练候选样本：{len(df_selected)} 行；"
        f"正样本 {int(df_selected['Label'].sum())}；"
        f"负样本 {len(df_selected) - int(df_selected['Label'].sum())}"
    )

    selected_path = OUTPUT_DIR / "selected_training_samples_before_fault.csv"
    df_selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(f"已保存筛选样本：{selected_path}")

    return df_selected, x_col, y_col


# ============================================================
# 5. 断层特征计算
# ============================================================

def axial_mean_angle(angles, weights=None):
    """断层走向为轴向数据，0° 和 180° 等价。"""
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        weights = np.ones_like(angles)
    else:
        weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    if valid.sum() == 0:
        return FAULT_DIRECTION_FALLBACK, 0.0

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

    angles = []
    lengths = []

    if line is None or line.is_empty:
        return angles, lengths

    if isinstance(line, LineString):
        geoms = [line]
    elif isinstance(line, MultiLineString):
        geoms = list(line.geoms)
    else:
        try:
            geoms = [g for g in line.geoms if isinstance(g, LineString)]
        except Exception:
            return angles, lengths

    for geom in geoms:
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

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
    arr = arr.reshape(-1)
    out = []
    for v in arr:
        try:
            out.append(int(v))
        except Exception:
            pass
    return out


def compute_fault_features(df, x_col, y_col):
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as e:
        raise ImportError(
            "缺少 GIS 依赖，请先运行：\n"
            r"D:\anaconda\python.exe -m pip install geopandas shapely pyproj openpyxl"
        ) from e

    if not os.path.exists(FAULT_PATH):
        raise FileNotFoundError(f"找不到断裂带文件：{FAULT_PATH}")

    df = df.copy().reset_index(drop=True)

    raw_x = pd.to_numeric(df[x_col], errors="coerce")
    raw_y = pd.to_numeric(df[y_col], errors="coerce")

    crs_in = "EPSG:4326" if looks_like_lonlat(raw_x, raw_y) else None

    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(raw_x, raw_y)],
        crs=crs_in,
    )

    if points_gdf.crs is None:
        print("警告：点数据没有 CRS，脚本会把 x/y 当成平面坐标。")
        points_metric = points_gdf
        metric_crs = None
    else:
        metric_crs = choose_metric_crs(points_gdf)
        points_metric = points_gdf.to_crs(metric_crs)

    coords_metric = np.column_stack([
        points_metric.geometry.x.values,
        points_metric.geometry.y.values,
    ]).astype(np.float32)

    print(f"读取断裂带文件：{FAULT_PATH}")
    faults = gpd.read_file(FAULT_PATH)
    if faults.empty:
        raise ValueError("断裂带 shp 为空。")

    faults = faults[~faults.geometry.isna()].copy()
    faults = faults[~faults.geometry.is_empty].copy()

    if points_metric.crs is None:
        faults_metric = faults.copy()
    else:
        if faults.crs is None:
            print("警告：断裂带 shp 没有 CRS，假定它与点数据原始 CRS 一致。")
            faults = faults.set_crs(points_gdf.crs, allow_override=True)
        faults_metric = faults.to_crs(points_metric.crs)

    faults_metric = faults_metric.explode(index_parts=False).reset_index(drop=True)
    if faults_metric.empty:
        raise ValueError("断裂带 explode 后为空，请检查 shp 几何。")

    sindex = faults_metric.sindex

    n = len(points_metric)
    dist_fault = np.zeros(n, dtype=np.float32)
    fault_density = np.zeros(n, dtype=np.float32)
    local_angle = np.zeros(n, dtype=np.float32)
    local_confidence = np.zeros(n, dtype=np.float32)

    print("计算断层特征中... 只对筛选后的训练候选样本计算，不是对 60 万行全部计算。")

    for i, pt in enumerate(points_metric.geometry):
        # 最近距离：用全量 distance，更稳。筛选后样本量不大，速度可接受。
        distances = faults_metric.geometry.distance(pt)
        nearest_label = distances.idxmin()
        nearest_pos = int(faults_metric.index.get_loc(nearest_label))
        nearest_dist = distances.loc[nearest_label]
        if hasattr(nearest_dist, "iloc"):
            nearest_dist = nearest_dist.iloc[0]
        dist_fault[i] = float(nearest_dist)

        # 局部缓冲区内断层
        buf = pt.buffer(FAULT_SEARCH_RADIUS_M)
        try:
            cand_idx = normalize_sindex_result(sindex.query(buf, predicate="intersects"))
        except Exception:
            cand_idx = []

        cand_idx = [j for j in cand_idx if 0 <= j < len(faults_metric)]
        if len(cand_idx) == 0:
            cand_idx = [nearest_pos]

        cand = faults_metric.iloc[cand_idx].copy()

        if len(cand) > FAULT_MAX_NEAR_LINES:
            cand["_d"] = cand.geometry.distance(pt)
            cand = cand.nsmallest(FAULT_MAX_NEAR_LINES, "_d")

        try:
            local_len = float(cand.geometry.intersection(buf).length.sum())
        except Exception:
            local_len = float(cand.geometry.length.sum())

        area = math.pi * FAULT_SEARCH_RADIUS_M ** 2
        fault_density[i] = local_len / (area + 1e-12)

        angles_all = []
        lengths_all = []
        for geom in cand.geometry:
            a, l = line_orientation_and_length(geom)
            angles_all.extend(a)
            lengths_all.extend(l)

        theta, conf = axial_mean_angle(angles_all, lengths_all)
        local_angle[i] = theta
        local_confidence[i] = conf

        if (i + 1) % 1000 == 0:
            print(f"  已处理 {i + 1}/{n} 个样本")

    df["dist_fault"] = dist_fault
    df["fault_density"] = fault_density
    df["local_fault_angle"] = local_angle
    df["local_fault_confidence"] = local_confidence
    df["metric_x"] = coords_metric[:, 0]
    df["metric_y"] = coords_metric[:, 1]

    out_path = OUTPUT_DIR / "selected_training_samples_with_fault.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已保存断层特征样本：{out_path}")

    return df


# ============================================================
# 6. 特征矩阵与各向异性距离
# ============================================================

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

    return X_train, y_train, coords_train, theta_train, X_val, y_val, coords_val, theta_val


def compute_global_beta_ridge(X, y, alpha=1e-3):
    xtx = X.t().matmul(X)
    eye = torch.eye(xtx.size(0), device=X.device, dtype=X.dtype)
    xty = X.t().matmul(y)
    beta = torch.linalg.solve(xtx + alpha * eye, xty)
    return beta.view(-1)


def make_global_grid(coords_train, h=GRID_H, w=GRID_W, padding_ratio=0.05, device=DEVICE):
    xmin = coords_train[:, 0].min().item()
    xmax = coords_train[:, 0].max().item()
    ymin = coords_train[:, 1].min().item()
    ymax = coords_train[:, 1].max().item()

    xpad = (xmax - xmin) * padding_ratio + 1e-6
    ypad = (ymax - ymin) * padding_ratio + 1e-6

    xs = torch.linspace(xmin - xpad, xmax + xpad, steps=h, device=device)
    ys = torch.linspace(ymin - ypad, ymax + ypad, steps=w, device=device)

    grid_x = xs.view(h, 1).expand(h, w)
    grid_y = ys.view(1, w).expand(h, w)
    return grid_x, grid_y


def generate_anisotropic_grid(grid_x, grid_y, target_xy, theta):
    """
    对每个样本，计算从样本点到全局网格的方向距离，并按局部断层角旋转。
    输出 [B, 2, H, W]。
    """
    target_x = target_xy[:, 0].view(-1, 1, 1)
    target_y = target_xy[:, 1].view(-1, 1, 1)
    theta = theta.view(-1, 1, 1)

    du = grid_x.unsqueeze(0) - target_x
    dv = grid_y.unsqueeze(0) - target_y

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    d_parallel = du * cos_t + dv * sin_t
    d_perp = -du * sin_t + dv * cos_t

    # 沿断层方向除以更大的尺度，相当于“沿断层方向等效距离更近”。
    d_parallel = d_parallel / (ANISO_PARALLEL_SCALE * DISTANCE_NORMALIZE_M)
    d_perp = d_perp / (ANISO_PERP_SCALE * DISTANCE_NORMALIZE_M)

    aniso = torch.stack([d_parallel, d_perp], dim=1)
    aniso = torch.clamp(aniso, min=-20.0, max=20.0)
    return aniso


# ============================================================
# 7. 单折训练与评估
# ============================================================

def train_one_fold(fold, X_train, y_train, coords_train, theta_train, X_val, y_val, coords_val, theta_val):
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    coords_train = coords_train.to(DEVICE)
    theta_train = theta_train.to(DEVICE)

    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)
    coords_val = coords_val.to(DEVICE)
    theta_val = theta_val.to(DEVICE)

    beta_global = compute_global_beta_ridge(X_train, y_train).to(DEVICE)
    model = ACAWLR(coef_num=X_train.size(1), beta_global=beta_global).to(DEVICE)

    n_pos = float(y_train.sum().item())
    n_neg = float(y_train.numel() - y_train.sum().item())
    pos_weight_value = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_ds = TensorDataset(coords_train, theta_train, X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    grid_x, grid_y = make_global_grid(coords_train, GRID_H, GRID_W, device=DEVICE)

    best_val_ap = -1.0
    best_state = None
    patience_counter = 0

    print(f"\n========== Fold {fold} 开始 ==========")
    print(f"训练正样本：{int(n_pos)}，训练负样本：{int(n_neg)}，pos_weight={pos_weight_value:.3f}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for batch_coords, batch_theta, batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            aniso = generate_anisotropic_grid(grid_x, grid_y, batch_coords, batch_theta)
            logits = model(aniso, batch_x)
            loss = criterion(logits, batch_y)

            if not torch.isfinite(loss):
                print("警告：出现非有限 loss，跳过当前 batch。")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        avg_train_loss = total_loss / max(total_n, 1)

        val_prob, val_label = predict_prob(model, grid_x, grid_y, coords_val, theta_val, X_val, y_val)
        val_ap = safe_average_precision(val_label, val_prob)
        val_auc = safe_roc_auc(val_label, val_prob)

        print(
            f"Fold {fold} | Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={avg_train_loss:.6f} | val_AP={val_ap:.4f} | val_AUC={val_auc:.4f}"
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Fold {fold} early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_prob, val_label = predict_prob(model, grid_x, grid_y, coords_val, theta_val, X_val, y_val)
    metrics = compute_metrics(val_label, val_prob)
    metrics["fold"] = fold
    metrics["best_val_ap"] = best_val_ap

    model_path = OUTPUT_DIR / f"model_fold_{fold}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "beta_global": beta_global.detach().cpu(),
            "fold": fold,
        },
        model_path,
    )
    print(f"Fold {fold} 模型已保存：{model_path}")

    return metrics, val_prob, val_label


@torch.no_grad()
def predict_prob(model, grid_x, grid_y, coords, theta, X, y=None):
    model.eval()
    ds = TensorDataset(coords, theta, X) if y is None else TensorDataset(coords, theta, X, y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    probs = []
    labels = []

    for batch in loader:
        if y is None:
            batch_coords, batch_theta, batch_x = batch
            batch_y = None
        else:
            batch_coords, batch_theta, batch_x, batch_y = batch

        aniso = generate_anisotropic_grid(grid_x, grid_y, batch_coords, batch_theta)
        logits = model(aniso, batch_x)
        prob = torch.sigmoid(logits).view(-1)
        probs.append(prob.detach().cpu().numpy())

        if batch_y is not None:
            labels.append(batch_y.view(-1).detach().cpu().numpy())

    probs = np.concatenate(probs)
    if y is None:
        return probs, None

    labels = np.concatenate(labels)
    return probs, labels


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


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
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
        "threshold": threshold,
        "n_val": int(len(y_true)),
        "n_val_pos": int(y_true.sum()),
        "n_val_neg": int(len(y_true) - y_true.sum()),
    }
    return out


# ============================================================
# 8. 五折交叉验证主流程
# ============================================================

def run_fivefold_validation(df):
    feature_cols = BASE_FEATURE_COLS + [
        "dist_fault",
        "fault_density",
        "local_fault_confidence",
        # 坐标也作为弱空间位置特征进入回归项，帮助模型表达空间非平稳性。
        "metric_x",
        "metric_y",
    ]

    y = df["Label"].values.astype(int)

    n_pos = int(y.sum())
    n_splits = min(N_SPLITS, n_pos)
    if n_splits < 2:
        raise ValueError("正样本太少，无法做交叉验证。")

    if n_splits < N_SPLITS:
        print(f"正样本数不足，折数从 {N_SPLITS} 调整为 {n_splits}。")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    all_metrics = []
    cv_pred_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        (
            X_train,
            y_train,
            coords_train,
            theta_train,
            X_val,
            y_val,
            coords_val,
            theta_val,
        ) = prepare_features_for_fold(df, train_idx, val_idx, feature_cols)

        metrics, val_prob, val_label = train_one_fold(
            fold,
            X_train,
            y_train,
            coords_train,
            theta_train,
            X_val,
            y_val,
            coords_val,
            theta_val,
        )

        all_metrics.append(metrics)

        val_df = df.iloc[val_idx].copy().reset_index(drop=True)
        val_df["fold"] = fold
        val_df["cv_prob"] = val_prob
        val_df["cv_label"] = val_label.astype(int)
        cv_pred_rows.append(val_df)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUTPUT_DIR / "cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    cv_pred_df = pd.concat(cv_pred_rows, ignore_index=True)
    pred_path = OUTPUT_DIR / "cv_predictions.csv"
    cv_pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print("\n========== Five-fold validation 汇总 ==========")
    print(metrics_df)
    print("\n平均指标：")
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    print(metrics_df[numeric_cols].mean())

    print(f"\n已保存指标：{metrics_path}")
    print(f"已保存交叉验证预测：{pred_path}")

    return metrics_df, cv_pred_df


# ============================================================
# 9. 主函数
# ============================================================

def main():
    print(f"当前设备：{DEVICE}")
    print("注意：本脚本不会把 60 万行全部用于训练，而是保留全部正样本并筛选部分负样本。")

    df_selected, x_col, y_col = load_and_select_samples()
    df_selected = compute_fault_features(df_selected, x_col, y_col)

    # 保存最终进入 five-fold 的样本
    final_selected_path = OUTPUT_DIR / "selected_training_samples_final.csv"
    df_selected.to_csv(final_selected_path, index=False, encoding="utf-8-sig")
    print(f"最终训练候选样本已保存：{final_selected_path}")

    run_fivefold_validation(df_selected)

    print("\n全部流程结束。")


if __name__ == "__main__":
    main()

