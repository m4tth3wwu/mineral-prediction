# -*- coding: utf-8 -*-
"""
ACAWLR with fault constraints
--------------------------------
用途：
1. 读取地球化学点数据（CSV / XLSX）
2. 读取美国-加拿大断裂带矢量数据（SHP / GeoJSON / GPKG）
3. 自动计算：
   - dist_fault：到最近断层距离
   - fault_density：局部断层密度
   - local_fault_angle：局部主断层走向
4. 用 local_fault_angle 构建各向异性距离：
   - d_parallel：沿断层方向距离
   - d_perp：垂直断层方向距离
5. 训练 ACAWLR 简化实现版本

重要提醒：
- 点数据和断裂带数据必须在同一地理区域。
- 如果点坐标是经纬度，本脚本会自动投影到米制坐标系。
- 需要安装 geopandas / shapely / pyproj：
  pip install geopandas shapely pyproj openpyxl
"""

import os
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 0. 配置区：你主要改这里
# ============================================================

DATA_PATH = r"D:\code\ResearchPractice\data\geochemistry\沉积物地球化学数据.csv"

# 断裂带 shp 路径：注意只需要填 .shp 文件路径，其余 .dbf/.shx/.prj 放在同一文件夹即可
FAULT_PATH = r"D:\code\ResearchPractice\data\faults\GeologyFaults_USCanada.shp"

# 你的表格列名
CU_COL = "Cu_ppm"
COORD_COLS_CANDIDATES = [
    ("x", "y"),
    ("longitude", "latitude"),
    ("lon", "lat"),
    ("Longitude", "Latitude"),
]
BASE_FEATURE_COLS = ["Au_ppm", "Mo_ppm", "Fe_pct"]

# Cu 阈值：大于等于该值标记为正样本
CU_THRESHOLD = 3000

# fault 特征参数
FAULT_SEARCH_RADIUS_M = 50_000.0      # 计算局部断层方向、断层密度的半径，单位 m
FAULT_MAX_NEAR_LINES = 80             # 每个点最多使用附近多少条断层线，防止太慢
FAULT_DIRECTION_FALLBACK = 0.0        # 找不到附近断层时默认方向，单位弧度

# 各向异性参数
# 沿断层方向影响范围更大，垂直方向影响范围更小
ANISO_PARALLEL_SCALE = 3.0
ANISO_PERP_SCALE = 1.0
DISTANCE_NORMALIZE_M = 50_000.0

# GASPG 网格大小
GRID_H = 120
GRID_W = 100

# 训练参数
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出文件
OUTPUT_DIR = Path("./acawlr_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. 模型结构
# ============================================================

class ASPNN(nn.Module):
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
        ca_out = self.ca_sigmoid(avg_out + max_out)
        x = x * ca_out

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_out = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * sa_out


class CAWNN(nn.Module):
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
    def __init__(self, coef_num, beta_ols):
        super().__init__()
        self.aspnn = ASPNN()
        self.cawnn = CAWNN(coef_num)
        self.register_buffer("beta_ols", beta_ols.view(-1))

    def forward(self, aniso_distances, x_features):
        """
        aniso_distances: [B, 2, H, W]
        x_features: [B, P]，其中 P = 特征数 + 1 个截距项
        输出：logit [B, 1]
        """
        b, c, h, w = aniso_distances.shape
        if c != 2:
            raise ValueError(f"aniso_distances 的通道数必须是 2，但收到 {c}")

        d_reshaped = aniso_distances.permute(0, 2, 3, 1).reshape(-1, 2)
        proximity_weights = self.aspnn(d_reshaped)
        gaspg = proximity_weights.view(b, 1, h, w)
        gaspg = torch.clamp(gaspg, min=1e-6, max=1.0 - 1e-6)

        w_k = self.cawnn(gaspg)
        w_k = torch.clamp(w_k, min=-50.0, max=50.0)

        beta_local = w_k * self.beta_ols.view(1, -1)
        z = torch.sum(beta_local * x_features, dim=1, keepdim=True)
        z = torch.clamp(z, min=-30.0, max=30.0)
        return z


# ============================================================
# 2. 基础工具函数
# ============================================================

def smart_read_table(path):
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if suffix == ".csv":
        # 兼容中文路径和常见编码
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)

    raise ValueError(f"暂不支持的数据格式：{suffix}，请使用 CSV 或 Excel。")


def infer_coord_cols(df):
    for x_col, y_col in COORD_COLS_CANDIDATES:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    raise ValueError(
        f"找不到坐标列。请确认表格中至少有这些组合之一：{COORD_COLS_CANDIDATES}"
    )


def looks_like_lonlat(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        np.nanmin(x) >= -180 and np.nanmax(x) <= 180 and
        np.nanmin(y) >= -90 and np.nanmax(y) <= 90
    )


def choose_metric_crs(points_gdf):
    """
    自动选择米制投影。
    对小范围数据，优先用 UTM；
    对大范围北美数据，尝试 North America Albers；
    失败时退回 EPSG:3857。
    """
    try:
        bounds = points_gdf.total_bounds
        lon_span = bounds[2] - bounds[0]
        lat_span = bounds[3] - bounds[1]

        # 小区域用 UTM，距离和角度更合理
        if lon_span <= 12 and lat_span <= 12:
            utm = points_gdf.estimate_utm_crs()
            if utm is not None:
                return utm

        # 大范围北美，优先用 ESRI:102008
        try:
            from pyproj import CRS
            return CRS.from_user_input("ESRI:102008")
        except Exception:
            return "EPSG:3857"

    except Exception:
        return "EPSG:3857"


def compute_ols_ridge(X, y, alpha=1e-3):
    """
    原论文用 LR/OLS 作为全局项。
    这里保留你的原始思路，用 ridge 避免矩阵奇异。
    """
    xtx = X.t().matmul(X)
    eye = torch.eye(xtx.size(0), device=X.device, dtype=X.dtype)
    xty = X.t().matmul(y)
    return torch.linalg.solve(xtx + alpha * eye, xty).view(-1)


def axial_mean_angle(angles, weights=None):
    """
    断层走向是轴向数据：0° 和 180° 等价。
    所以用 2theta 求平均。
    """
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
    """
    返回 LineString / MultiLineString 的局部段方向与长度。
    """
    from shapely.geometry import LineString, MultiLineString

    angles = []
    lengths = []

    if line is None or line.is_empty:
        return angles, lengths

    geoms = []
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
            # 走向按 0-pi 处理
            if theta < 0:
                theta += math.pi
            if theta >= math.pi:
                theta -= math.pi
            angles.append(theta)
            lengths.append(length)

    return angles, lengths


# ============================================================
# 3. fault 特征提取
# ============================================================

def build_points_and_fault_features(df, x_col, y_col, fault_path=None):
    """
    返回：
    - df_aug: 增加 fault 特征后的 dataframe
    - coords_metric: 米制坐标 [N, 2]
    - metric_crs: 模型使用的投影坐标系
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as e:
        raise ImportError(
            "需要安装 geopandas / shapely / pyproj：\n"
            "pip install geopandas shapely pyproj openpyxl"
        ) from e

    df = df.copy()

    raw_x = pd.to_numeric(df[x_col], errors="coerce")
    raw_y = pd.to_numeric(df[y_col], errors="coerce")
    valid_coord = raw_x.notna() & raw_y.notna()
    if not valid_coord.all():
        print(f"坐标缺失：删除 {(~valid_coord).sum()} 行。")
        df = df.loc[valid_coord].reset_index(drop=True)
        raw_x = raw_x.loc[valid_coord].reset_index(drop=True)
        raw_y = raw_y.loc[valid_coord].reset_index(drop=True)

    crs_in = "EPSG:4326" if looks_like_lonlat(raw_x, raw_y) else None
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(raw_x, raw_y)],
        crs=crs_in,
    )

    # 如果是经纬度，自动转米制投影
    if points_gdf.crs is None:
        warnings.warn(
            "点数据没有 CRS，脚本会把 x/y 当成已经是米制平面坐标。"
            "如果你的 x/y 是经纬度，请确保列名和数值范围正确。"
        )
        points_metric = points_gdf
        metric_crs = None
    else:
        metric_crs = choose_metric_crs(points_gdf)
        points_metric = points_gdf.to_crs(metric_crs)

    coords_metric = np.column_stack([
        points_metric.geometry.x.values,
        points_metric.geometry.y.values,
    ]).astype(np.float32)

    # 默认 fault 特征
    df["dist_fault"] = np.nan
    df["fault_density"] = 0.0
    df["local_fault_angle"] = FAULT_DIRECTION_FALLBACK
    df["local_fault_confidence"] = 0.0

    if not fault_path:
        print("未提供 FAULT_PATH：跳过 fault 特征，后续将使用默认方向。")
        return df, coords_metric, metric_crs

    fault_path = str(fault_path)
    if not os.path.exists(fault_path):
        raise FileNotFoundError(f"找不到断裂带文件：{fault_path}")

    print(f"读取断裂带文件：{fault_path}")
    faults = gpd.read_file(fault_path)

    if faults.empty:
        raise ValueError("断裂带文件为空。")

    # 只保留线几何
    faults = faults[~faults.geometry.isna()].copy()
    faults = faults[~faults.geometry.is_empty].copy()

    if points_metric.crs is None:
        if faults.crs is not None:
            warnings.warn(
                "点数据没有 CRS，但断裂带有 CRS。请确认二者坐标单位一致。"
            )
        faults_metric = faults
    else:
        if faults.crs is None:
            warnings.warn(
                "断裂带文件没有 CRS，将假定它和点数据原始 CRS 一致。"
                "如果不一致，距离和方向会错。"
            )
            faults = faults.set_crs(points_gdf.crs, allow_override=True)
        faults_metric = faults.to_crs(points_metric.crs)

    # explode，便于空间索引
    faults_metric = faults_metric.explode(index_parts=False).reset_index(drop=True)
    sindex = faults_metric.sindex

    dist_fault = np.zeros(len(points_metric), dtype=np.float32)
    fault_density = np.zeros(len(points_metric), dtype=np.float32)
    local_angle = np.zeros(len(points_metric), dtype=np.float32)
    local_conf = np.zeros(len(points_metric), dtype=np.float32)

    print("计算 fault 特征中，可能需要一段时间...")

    total_fault_length = float(faults_metric.length.sum())
    if total_fault_length <= 0:
        raise ValueError("断裂带总长度为 0，请检查 shp 几何。")

    for idx, pt in enumerate(points_metric.geometry):
        # 最近断层距离
        try:
            nearest_idx = list(sindex.nearest(pt, 1))[0]
            nearest_geom = faults_metric.geometry.iloc[nearest_idx]
            dist_fault[idx] = pt.distance(nearest_geom)
        except Exception:
            # 兼容不同 geopandas/shapely 版本
            distances = faults_metric.geometry.distance(pt)
            nearest_idx = int(distances.idxmin())
            nearest_geom = faults_metric.geometry.loc[nearest_idx]
            dist_fault[idx] = float(distances.loc[nearest_idx])

        # 局部断层：搜索半径内
        buf = pt.buffer(FAULT_SEARCH_RADIUS_M)
        candidate_idx = list(sindex.query(buf, predicate="intersects"))

        if len(candidate_idx) == 0:
            # 找不到附近断层，用最近断层方向
            candidate_idx = [nearest_idx]

        if len(candidate_idx) > FAULT_MAX_NEAR_LINES:
            # 候选太多时，取离点最近的若干条，避免极慢
            cand = faults_metric.iloc[candidate_idx].copy()
            cand["_d"] = cand.geometry.distance(pt)
            cand = cand.nsmallest(FAULT_MAX_NEAR_LINES, "_d")
        else:
            cand = faults_metric.iloc[candidate_idx].copy()

        # fault density = 半径内断层总长度 / 圆面积
        try:
            clipped_lengths = cand.geometry.intersection(buf).length
            local_len = float(clipped_lengths.sum())
        except Exception:
            local_len = float(cand.geometry.length.sum())

        area = math.pi * FAULT_SEARCH_RADIUS_M ** 2
        fault_density[idx] = local_len / (area + 1e-12)

        # 局部主断层方向：按线段长度加权
        angles_all = []
        lengths_all = []
        for geom in cand.geometry:
            a, l = line_orientation_and_length(geom)
            angles_all.extend(a)
            lengths_all.extend(l)

        theta, conf = axial_mean_angle(angles_all, lengths_all)
        local_angle[idx] = theta
        local_conf[idx] = conf

        if (idx + 1) % 5000 == 0:
            print(f"  已处理 {idx + 1}/{len(points_metric)} 个点")

    df["dist_fault"] = dist_fault
    df["fault_density"] = fault_density
    df["local_fault_angle"] = local_angle
    df["local_fault_confidence"] = local_conf

    print("fault 特征计算完成。")
    return df, coords_metric, metric_crs


# ============================================================
# 4. 数据处理
# ============================================================

def data_processing(data_path, fault_path=None):
    print("读取点数据...")
    df = smart_read_table(data_path)
    print(f"原始数据量：{len(df)} 行")
    print(f"字段：{list(df.columns)}")

    x_col, y_col = infer_coord_cols(df)
    print(f"使用坐标列：x = {x_col}, y = {y_col}")

    needed = [CU_COL, x_col, y_col] + BASE_FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{missing}")

    # 数值化
    for c in [CU_COL, x_col, y_col] + BASE_FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 删除坐标缺失和控矿特征全缺失行
    df = df.dropna(subset=[x_col, y_col]).reset_index(drop=True)

    # 先计算 fault 特征与投影坐标
    df, coords_metric, metric_crs = build_points_and_fault_features(
        df, x_col, y_col, fault_path=fault_path
    )

    feature_cols = BASE_FEATURE_COLS + [
        "dist_fault",
        "fault_density",
        "local_fault_confidence",
    ]

    # local_fault_angle 不作为普通特征直接进入逻辑回归，而是用于构建各向异性距离
    # 这样可以避免角度 0 和 pi 的边界问题

    for c in feature_cols + [CU_COL, "local_fault_angle"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 训练集：Cu 有值的点
    train_mask = df[CU_COL].notna()
    df_train = df.loc[train_mask].copy().reset_index(drop=True)
    df_pred = df.loc[~train_mask].copy().reset_index(drop=True)

    if df_train.empty:
        raise ValueError("没有可训练样本：Cu_ppm 全部为空。")

    # 标签
    df_train["Label"] = (df_train[CU_COL] >= CU_THRESHOLD).astype(np.float32)

    print(
        f"训练样本数：{len(df_train)}；正样本数：{int(df_train['Label'].sum())}；"
        f"待预测样本数：{len(df_pred)}"
    )

    # 缺失值填充：用训练集 median
    medians = df_train[feature_cols].median(numeric_only=True)
    df_train[feature_cols] = df_train[feature_cols].fillna(medians)
    if not df_pred.empty:
        df_pred[feature_cols] = df_pred[feature_cols].fillna(medians)

    # 标准化：只用训练集统计量
    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1.0)

    df_train[feature_cols] = (df_train[feature_cols] - means) / stds
    if not df_pred.empty:
        df_pred[feature_cols] = (df_pred[feature_cols] - means) / stds

    # 对应的米制坐标和角度
    coords_train = coords_metric[train_mask.values]
    theta_train = df_train["local_fault_angle"].values.astype(np.float32)

    X_raw_train = torch.tensor(df_train[feature_cols].values, dtype=torch.float32)
    intercept = torch.ones(X_raw_train.size(0), 1, dtype=torch.float32)
    X_train = torch.cat([X_raw_train, intercept], dim=1)
    y_train = torch.tensor(df_train["Label"].values, dtype=torch.float32).view(-1, 1)
    coords_train_t = torch.tensor(coords_train, dtype=torch.float32)
    theta_train_t = torch.tensor(theta_train, dtype=torch.float32).view(-1, 1)

    pred_bundle = None
    if not df_pred.empty:
        coords_pred = coords_metric[~train_mask.values]
        theta_pred = df_pred["local_fault_angle"].values.astype(np.float32)

        X_raw_pred = torch.tensor(df_pred[feature_cols].values, dtype=torch.float32)
        intercept_pred = torch.ones(X_raw_pred.size(0), 1, dtype=torch.float32)
        X_pred = torch.cat([X_raw_pred, intercept_pred], dim=1)

        pred_bundle = {
            "df_pred": df_pred,
            "coords_pred": torch.tensor(coords_pred, dtype=torch.float32),
            "theta_pred": torch.tensor(theta_pred, dtype=torch.float32).view(-1, 1),
            "X_pred": X_pred,
        }

    train_bundle = {
        "df_train": df_train,
        "coords_train": coords_train_t,
        "theta_train": theta_train_t,
        "X_train": X_train,
        "y_train": y_train,
        "feature_cols": feature_cols,
        "metric_crs": metric_crs,
        "standardization_means": means,
        "standardization_stds": stds,
    }

    # 保存带 fault 特征的数据，便于检查
    aug_path = OUTPUT_DIR / "data_with_fault_features.csv"
    df.to_csv(aug_path, index=False, encoding="utf-8-sig")
    print(f"已保存 fault 特征检查表：{aug_path}")

    return train_bundle, pred_bundle


# ============================================================
# 5. 各向异性网格生成
# ============================================================

def make_model_grid(coords_train, h=GRID_H, w=GRID_W, padding_ratio=0.05, device=DEVICE):
    """
    根据训练点的米制坐标自动生成研究区网格。
    """
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


def generate_fault_anisotropic_grid(grid_x, grid_y, target_xy, theta):
    """
    grid_x/grid_y: [H, W]
    target_xy: [B, 2]
    theta: [B, 1]，局部断层走向，弧度，0-pi
    return: [B, 2, H, W]
    """
    target_x = target_xy[:, 0].view(-1, 1, 1)
    target_y = target_xy[:, 1].view(-1, 1, 1)
    theta = theta.view(-1, 1, 1)

    du = grid_x.unsqueeze(0) - target_x
    dv = grid_y.unsqueeze(0) - target_y

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # 沿断层方向
    d_parallel = du * cos_t + dv * sin_t

    # 垂直断层方向
    d_perp = -du * sin_t + dv * cos_t

    # 各向异性缩放：沿断层方向“变近”，垂直方向“变远”
    d_parallel = d_parallel / (ANISO_PARALLEL_SCALE * DISTANCE_NORMALIZE_M)
    d_perp = d_perp / (ANISO_PERP_SCALE * DISTANCE_NORMALIZE_M)

    aniso = torch.stack([d_parallel, d_perp], dim=1)
    aniso = torch.clamp(aniso, min=-20.0, max=20.0)
    return aniso


# ============================================================
# 6. 训练与预测
# ============================================================

def train_model(train_bundle):
    coords_train = train_bundle["coords_train"].to(DEVICE)
    theta_train = train_bundle["theta_train"].to(DEVICE)
    X = train_bundle["X_train"].to(DEVICE)
    y = train_bundle["y_train"].to(DEVICE)

    # 清理 NaN
    valid = (
        torch.isfinite(coords_train).all(dim=1) &
        torch.isfinite(theta_train).all(dim=1).view(-1) &
        torch.isfinite(X).all(dim=1) &
        torch.isfinite(y).all(dim=1)
    )

    coords_train = coords_train[valid]
    theta_train = theta_train[valid]
    X = X[valid]
    y = y[valid]

    print(f"有效训练样本数：{X.size(0)}")

    beta_ols = compute_ols_ridge(X, y).to(DEVICE)
    print("全局 beta_ols：", beta_ols.detach().cpu().numpy())

    model = ACAWLR(coef_num=X.size(1), beta_ols=beta_ols).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(coords_train, theta_train, X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    grid_x, grid_y = make_model_grid(coords_train, GRID_H, GRID_W, device=DEVICE)

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_n = 0

        for batch_coords, batch_theta, batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            aniso_dist = generate_fault_anisotropic_grid(
                grid_x, grid_y, batch_coords, batch_theta
            )

            logits = model(aniso_dist, batch_x)
            loss = criterion(logits, batch_y)

            if not torch.isfinite(loss):
                print("警告：当前 batch loss 非有限，已跳过。")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        avg_loss = total_loss / max(total_n, 1)
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] | Average Loss = {avg_loss:.6f}")

    model_path = OUTPUT_DIR / "acawlr_fault_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "beta_ols": beta_ols.detach().cpu(),
            "feature_cols": train_bundle["feature_cols"],
            "config": {
                "GRID_H": GRID_H,
                "GRID_W": GRID_W,
                "CU_THRESHOLD": CU_THRESHOLD,
                "FAULT_SEARCH_RADIUS_M": FAULT_SEARCH_RADIUS_M,
                "ANISO_PARALLEL_SCALE": ANISO_PARALLEL_SCALE,
                "ANISO_PERP_SCALE": ANISO_PERP_SCALE,
                "DISTANCE_NORMALIZE_M": DISTANCE_NORMALIZE_M,
            },
        },
        model_path,
    )
    print(f"模型已保存：{model_path}")

    return model, grid_x, grid_y


@torch.no_grad()
def predict_for_bundle(model, grid_x, grid_y, coords, theta, X, batch_size=BATCH_SIZE):
    model.eval()

    coords = coords.to(DEVICE)
    theta = theta.to(DEVICE)
    X = X.to(DEVICE)

    scores = []

    dataset = TensorDataset(coords, theta, X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    for batch_coords, batch_theta, batch_x in loader:
        aniso_dist = generate_fault_anisotropic_grid(grid_x, grid_y, batch_coords, batch_theta)
        logits = model(aniso_dist, batch_x)
        prob = torch.sigmoid(logits).view(-1)
        scores.append(prob.detach().cpu())

    return torch.cat(scores).numpy()


def main():
    train_bundle, pred_bundle = data_processing(DATA_PATH, fault_path=FAULT_PATH)
    model, grid_x, grid_y = train_model(train_bundle)

    # 训练集也输出一个拟合概率，方便你检查效果
    train_scores = predict_for_bundle(
        model,
        grid_x,
        grid_y,
        train_bundle["coords_train"],
        train_bundle["theta_train"],
        train_bundle["X_train"],
    )

    df_train_out = train_bundle["df_train"].copy()
    df_train_out["ACAWLR_score"] = train_scores
    train_out_path = OUTPUT_DIR / "train_scores.csv"
    df_train_out.to_csv(train_out_path, index=False, encoding="utf-8-sig")
    print(f"训练集评分已保存：{train_out_path}")

    if pred_bundle is not None:
        pred_scores = predict_for_bundle(
            model,
            grid_x,
            grid_y,
            pred_bundle["coords_pred"],
            pred_bundle["theta_pred"],
            pred_bundle["X_pred"],
        )

        df_pred_out = pred_bundle["df_pred"].copy()
        df_pred_out["ACAWLR_score"] = pred_scores
        pred_out_path = OUTPUT_DIR / "prediction_scores.csv"
        df_pred_out.to_csv(pred_out_path, index=False, encoding="utf-8-sig")
        print(f"预测结果已保存：{pred_out_path}")

    print("全部流程结束。")


if __name__ == "__main__":
    main()

