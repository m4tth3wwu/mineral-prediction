"""
ACAWLR_fault_ready.py

相对你原始代码的主要改动：
1. 支持接入断裂带数据 fault：CSV / Shapefile / GeoJSON。
2. 自动计算 fault 特征：dist_fault、fault_density、local_fault_angle。
3. 用 local_fault_angle 构建沿断层/垂直断层的各向异性距离，而不是只依赖矿点 PCA。
4. 自动根据训练坐标生成 grid，不再写死 100-105、30-35。
5. 使用 DataLoader 批量训练，不再一个样本一个样本循环。
6. 如果没有 fault 文件，会退回到“矿点 PCA 全局方向”。

断裂带 CSV 推荐字段：
    x1, y1, x2, y2
可选字段：
    weight 或 level

重要提醒：
    如果 x/y 是经纬度，经纬度不适合直接算距离和角度。
    最好先把点数据和断裂带数据统一投影到同一个平面坐标系，单位 m 或 km。
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =========================
# 1. 神经网络模块
# =========================

class ASPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.ca_sigmoid = nn.Sigmoid()
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.mlp(nn.AdaptiveAvgPool2d(1)(x))
        max_out = self.mlp(nn.AdaptiveMaxPool2d(1)(x))
        x = x * self.ca_sigmoid(avg_out + max_out)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([avg_out, max_out], dim=1))


class CAWNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.head = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.res1 = ResBlock(32, 8)
        self.res2 = ResBlock(8, 16)
        self.res3 = ResBlock(16, 32)
        self.cbam = CBAM(32)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, out_dim)

    def forward(self, gaspg):
        x = self.head(gaspg)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.cbam(x)
        return self.fc(self.avgpool(x).view(x.size(0), -1))


class ACAWLR(nn.Module):
    def __init__(self, x_dim, beta_ols):
        super().__init__()
        self.aspnn = ASPNN()
        self.cawnn = CAWNN(out_dim=x_dim)
        self.register_buffer("beta_ols", beta_ols.view(-1))

    def forward(self, aniso_distances, x_features):
        """
        aniso_distances: [B, 2, H, W]
            第 1 个通道：沿局部断层方向的距离分量 d_parallel
            第 2 个通道：垂直局部断层方向的距离分量 d_perp
        x_features: [B, P]
            P 个特征，最后一列通常是截距项。
        """
        batch_size, _, H, W = aniso_distances.shape

        d_reshaped = aniso_distances.permute(0, 2, 3, 1).reshape(-1, 2)
        proximity_weights = self.aspnn(d_reshaped)
        gaspg = proximity_weights.view(batch_size, 1, H, W)
        gaspg = torch.clamp(gaspg, min=1e-6, max=1.0 - 1e-6)

        w_k = self.cawnn(gaspg)
        w_k = torch.clamp(w_k, min=-50.0, max=50.0)

        beta_local = w_k * self.beta_ols.view(1, -1)
        z = torch.sum(beta_local * x_features, dim=1, keepdim=True)
        z = torch.clamp(z, min=-30.0, max=30.0)
        return z


# =========================
# 2. 数学与空间工具函数
# =========================

def compute_ols(X, y, alpha=1e-3):
    """岭稳定版 OLS，用作全局 beta 初值。"""
    XtX = X.t().matmul(X)
    identity = torch.eye(XtX.size(0), device=X.device, dtype=X.dtype)
    beta_ols = torch.linalg.solve(XtX + alpha * identity, X.t().matmul(y))
    return beta_ols.view(-1)


def compute_anisotropy_basis(deposit_coords, weights=None):
    """
    fallback：没有 fault 数据时，用正样本矿点 PCA 估计全局主方向。
    deposit_coords: [N, 2]
    weights: [N, 1]
    """
    N = deposit_coords.size(0)
    if N < 2:
        return torch.eye(2, device=deposit_coords.device, dtype=deposit_coords.dtype)

    if weights is None:
        weights = torch.ones(N, 1, device=deposit_coords.device, dtype=deposit_coords.dtype)

    weight_sum = torch.sum(weights).clamp_min(1e-12)
    mean_coords = torch.sum(deposit_coords * weights, dim=0) / weight_sum
    centered_coords = deposit_coords - mean_coords

    W = torch.diag(weights.squeeze())
    cov_matrix = centered_coords.t().matmul(W).matmul(centered_coords) / max(N - 1, 1)
    _, eigenvectors = torch.linalg.eigh(cov_matrix)
    basis = torch.flip(eigenvectors, dims=[1])
    return basis


def basis_to_angle(basis: torch.Tensor) -> float:
    """取 PCA 第一主轴方向角。"""
    e1 = basis[:, 0]
    return float(torch.atan2(e1[1], e1[0]).item())


def normalize_coords_same_scale(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    用同一个尺度缩放 x 和 y，避免改变方向角。
    不要分别标准化 x/y，否则断层角度会被扭曲。
    """
    center = np.nanmean(coords, axis=0)
    scale = float(np.nanstd(coords))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (coords - center) / scale, center, scale


def apply_coord_normalization(coords: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (coords - center) / scale


def make_grid_from_coords(coords_model: torch.Tensor, H=120, W=100, margin=0.05, device=None):
    """根据训练坐标自动生成模型空间网格。"""
    if device is None:
        device = coords_model.device

    x_min, x_max = coords_model[:, 0].min(), coords_model[:, 0].max()
    y_min, y_max = coords_model[:, 1].min(), coords_model[:, 1].max()

    dx = (x_max - x_min).clamp_min(1e-6)
    dy = (y_max - y_min).clamp_min(1e-6)
    x_min = x_min - margin * dx
    x_max = x_max + margin * dx
    y_min = y_min - margin * dy
    y_max = y_max + margin * dy

    grid_u = torch.linspace(x_min, x_max, steps=H, device=device).view(H, 1).expand(H, W)
    grid_v = torch.linspace(y_min, y_max, steps=W, device=device).view(1, W).expand(H, W)
    return grid_u, grid_v


def generate_fault_anisotropic_grid_batch(
    grid_u,
    grid_v,
    target_coords,
    target_angles,
    parallel_scale=3.0,
    perpendicular_scale=1.0,
):
    """
    用局部断层方向构建各向异性距离网格。

    target_angles: [B]
        每个样本点附近的局部断层走向角，单位 rad。

    parallel_scale > perpendicular_scale 表示：沿断层方向影响范围更长。
    """
    B = target_coords.size(0)
    H, W = grid_u.shape

    du = grid_u.unsqueeze(0) - target_coords[:, 0].view(B, 1, 1)
    dv = grid_v.unsqueeze(0) - target_coords[:, 1].view(B, 1, 1)

    cos_t = torch.cos(target_angles).view(B, 1, 1)
    sin_t = torch.sin(target_angles).view(B, 1, 1)

    d_parallel = du * cos_t + dv * sin_t
    d_perp = -du * sin_t + dv * cos_t

    d_parallel = d_parallel / max(float(parallel_scale), 1e-6)
    d_perp = d_perp / max(float(perpendicular_scale), 1e-6)

    aniso = torch.stack([d_parallel, d_perp], dim=1)
    return torch.clamp(aniso, min=-1e4, max=1e4)


# =========================
# 3. fault 数据读取与特征计算
# =========================

def _read_fault_from_vector_file(path: Path) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("读取 shp/geojson 需要安装 geopandas。也可以先把断层线转成 x1,y1,x2,y2 的 CSV。") from exc

    gdf = gpd.read_file(path)
    rows = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        weight = 1.0
        for col in ["weight", "Weight", "level", "Level", "rank", "Rank"]:
            if col in gdf.columns and pd.notna(row[col]):
                try:
                    value = float(row[col])
                    # level 越小通常越重要；这里做一个温和转换。
                    if "level" in col.lower() or "rank" in col.lower():
                        weight = 1.0 / max(value, 1.0)
                    else:
                        weight = value
                except Exception:
                    weight = 1.0
                break

        geoms = []
        if geom.geom_type == "LineString":
            geoms = [geom]
        elif geom.geom_type == "MultiLineString":
            geoms = list(geom.geoms)
        else:
            continue

        for line in geoms:
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                rows.append({
                    "x1": a[0], "y1": a[1],
                    "x2": b[0], "y2": b[1],
                    "weight": weight,
                })

    return pd.DataFrame(rows)


def load_fault_segments(fault_path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    读取断裂带线段。
    支持：
      1. CSV: 必须有 x1,y1,x2,y2；可选 weight/level。
      2. shp / geojson：需要 geopandas。
    """
    if fault_path is None or str(fault_path).strip() == "":
        return None

    path = Path(fault_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 fault 文件: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".shp", ".geojson", ".json", ".gpkg"]:
        df = _read_fault_from_vector_file(path)
    else:
        raise ValueError("fault_path 只支持 .csv / .shp / .geojson / .gpkg")

    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    candidates = {
        "x1": ["x1", "start_x", "x_start", "from_x"],
        "y1": ["y1", "start_y", "y_start", "from_y"],
        "x2": ["x2", "end_x", "x_end", "to_x"],
        "y2": ["y2", "end_y", "y_end", "to_y"],
    }
    for target, names in candidates.items():
        for name in names:
            if name in lower_cols:
                rename_map[lower_cols[name]] = target
                break
    df = df.rename(columns=rename_map)

    required = ["x1", "y1", "x2", "y2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"fault CSV 缺少字段 {missing}。至少需要 x1,y1,x2,y2。")

    if "weight" not in df.columns:
        if "level" in df.columns:
            df["weight"] = 1.0 / np.maximum(pd.to_numeric(df["level"], errors="coerce").fillna(1.0).values, 1.0)
        else:
            df["weight"] = 1.0

    df = df[["x1", "y1", "x2", "y2", "weight"]].copy()
    for c in ["x1", "y1", "x2", "y2", "weight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    dx = df["x2"].values - df["x1"].values
    dy = df["y2"].values - df["y1"].values
    length = np.sqrt(dx * dx + dy * dy)
    df = df[length > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("fault 文件没有有效线段。")

    return df


def compute_fault_features_for_points(
    coords_raw: np.ndarray,
    fault_segments: Optional[pd.DataFrame],
    density_radius: float,
    chunk_size: int = 512,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    对每个点计算：
      dist_fault: 到最近断层线段的距离
      fault_density: 半径 density_radius 内的断层长度加权密度
      local_fault_angle: 最近断层线段走向角，rad

    如果没有 fault_segments，返回 None, None。
    """
    if fault_segments is None:
        return None, None

    seg = fault_segments
    a = seg[["x1", "y1"]].values.astype(np.float64)
    b = seg[["x2", "y2"]].values.astype(np.float64)
    weight = seg["weight"].values.astype(np.float64)

    v = b - a
    vv = np.sum(v * v, axis=1)
    seg_len = np.sqrt(vv)
    seg_angle = np.arctan2(v[:, 1], v[:, 0])

    n = coords_raw.shape[0]
    min_dist = np.full(n, np.inf, dtype=np.float64)
    nearest_angle = np.zeros(n, dtype=np.float64)
    density = np.zeros(n, dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        p = coords_raw[start:end].astype(np.float64)

        # [B, S, 2]
        w = p[:, None, :] - a[None, :, :]
        t = np.sum(w * v[None, :, :], axis=2) / np.maximum(vv[None, :], 1e-12)
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :, :] + t[:, :, None] * v[None, :, :]
        diff = p[:, None, :] - proj
        dist = np.sqrt(np.sum(diff * diff, axis=2))

        idx = np.argmin(dist, axis=1)
        min_dist[start:end] = dist[np.arange(end - start), idx]
        nearest_angle[start:end] = seg_angle[idx]

        within = dist <= density_radius
        # 断层长度加权密度：半径内线段越长、权重越高，density 越大。
        density[start:end] = (within * (seg_len * weight)[None, :]).sum(axis=1) / (math.pi * density_radius ** 2)

    feat = pd.DataFrame({
        "dist_fault": min_dist,
        "fault_density": density,
        "local_fault_angle": nearest_angle,
    })
    return feat, nearest_angle


# =========================
# 4. 数据处理
# =========================

def data_procession(
    csv_path,
    fault_path: Optional[str] = None,
    batch_size=50000,
    cu_col="Cu_ppm",
    coord_cols=("x", "y"),
    base_feature_cols=("Au_ppm", "Mo_ppm", "Fe_pct"),
    cu_threshold=3000,
    fault_density_radius: Optional[float] = None,
):
    """
    返回 train_bundle, predict_bundle, meta。
    """
    print("数据载入中...")
    chunks = pd.read_csv(csv_path, chunksize=batch_size)

    train_list = []
    pred_list = []

    for _, chunk in enumerate(chunks):
        for c in [cu_col, *coord_cols, *base_feature_cols]:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        train_mask = chunk[cu_col].notna()
        chunk_train = chunk[train_mask].copy()
        chunk_pred = chunk[~train_mask].copy()

        if not chunk_train.empty:
            train_list.append(chunk_train)
        if not chunk_pred.empty:
            pred_list.append(chunk_pred)

    df_train = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    df_pred = pd.concat(pred_list, ignore_index=True) if pred_list else pd.DataFrame()
    print(f"载入完成 -> 训练样本数: {len(df_train)} | 待预测样本数: {len(df_pred)}")

    if df_train.empty:
        raise ValueError("训练集为空：Cu_ppm 全是缺失或列名不对。")

    # 丢掉无坐标样本
    df_train = df_train.dropna(subset=list(coord_cols)).reset_index(drop=True)
    if not df_pred.empty:
        df_pred = df_pred.dropna(subset=list(coord_cols)).reset_index(drop=True)

    coords_train_raw = df_train[list(coord_cols)].values.astype(np.float64)
    coords_pred_raw = df_pred[list(coord_cols)].values.astype(np.float64) if not df_pred.empty else np.empty((0, 2))

    # 自动给 fault density 设半径：默认研究区较大边长的 2%。
    if fault_density_radius is None:
        x_span = np.nanmax(coords_train_raw[:, 0]) - np.nanmin(coords_train_raw[:, 0])
        y_span = np.nanmax(coords_train_raw[:, 1]) - np.nanmin(coords_train_raw[:, 1])
        fault_density_radius = max(x_span, y_span) * 0.02
        if fault_density_radius <= 0 or not np.isfinite(fault_density_radius):
            fault_density_radius = 1.0

    # 读取 fault 并计算 fault 特征
    fault_segments = load_fault_segments(fault_path)
    use_fault = fault_segments is not None

    if use_fault:
        print(f"已读取断裂带线段数: {len(fault_segments)}")
        print(f"开始计算 fault 特征，density_radius = {fault_density_radius:.4f}")
        fault_feat_train, angle_train = compute_fault_features_for_points(
            coords_train_raw, fault_segments, density_radius=fault_density_radius
        )
        df_train = pd.concat([df_train.reset_index(drop=True), fault_feat_train], axis=1)

        if not df_pred.empty:
            fault_feat_pred, angle_pred = compute_fault_features_for_points(
                coords_pred_raw, fault_segments, density_radius=fault_density_radius
            )
            df_pred = pd.concat([df_pred.reset_index(drop=True), fault_feat_pred], axis=1)
        else:
            angle_pred = np.empty((0,), dtype=np.float64)

        feature_cols = list(base_feature_cols) + ["dist_fault", "fault_density"]
    else:
        print("未提供 fault 文件：将退回到矿点 PCA 全局方向。")
        angle_train = None
        angle_pred = None
        feature_cols = list(base_feature_cols)

    # 标签
    df_train["Label"] = (pd.to_numeric(df_train[cu_col], errors="coerce") >= cu_threshold).astype(np.int32)

    # 特征缺失值处理 + 标准化
    for c in feature_cols:
        df_train[c] = pd.to_numeric(df_train[c], errors="coerce")
        if not df_pred.empty:
            df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce")

    medians = df_train[feature_cols].median()
    df_train[feature_cols] = df_train[feature_cols].fillna(medians)
    if not df_pred.empty:
        df_pred[feature_cols] = df_pred[feature_cols].fillna(medians)

    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1.0)
    df_train[feature_cols] = (df_train[feature_cols] - means) / stds
    if not df_pred.empty:
        df_pred[feature_cols] = (df_pred[feature_cols] - means) / stds

    # 坐标用同一尺度归一化，避免角度被拉歪。
    coords_train_model_np, coord_center, coord_scale = normalize_coords_same_scale(coords_train_raw)
    coords_pred_model_np = apply_coord_normalization(coords_pred_raw, coord_center, coord_scale) if len(coords_pred_raw) else coords_pred_raw

    # X + 截距项
    X_raw_train = torch.from_numpy(df_train[feature_cols].values.astype(np.float32))
    intercept_train = torch.ones(X_raw_train.size(0), 1)
    X_train_full = torch.cat([X_raw_train, intercept_train], dim=1)
    y_train = torch.from_numpy(df_train["Label"].values.astype(np.float32)).unsqueeze(1)

    coords_train_model = torch.from_numpy(coords_train_model_np.astype(np.float32))
    coords_train_raw_t = torch.from_numpy(coords_train_raw.astype(np.float32))

    # 正样本矿点坐标，用模型坐标做 PCA fallback
    deposit_mask = df_train["Label"].values == 1
    deposit_coords_model = coords_train_model[deposit_mask]
    if len(deposit_coords_model) < 2:
        deposit_coords_model = coords_train_model

    if use_fault:
        theta_train = torch.from_numpy(angle_train.astype(np.float32))
    else:
        basis = compute_anisotropy_basis(deposit_coords_model)
        global_angle = basis_to_angle(basis)
        theta_train = torch.full((coords_train_model.size(0),), global_angle, dtype=torch.float32)

    train_bundle = {
        "coords_train_model": coords_train_model,
        "coords_train_raw": coords_train_raw_t,
        "deposit_coords_model": deposit_coords_model,
        "X_train": X_train_full,
        "y_train": y_train,
        "theta_train": theta_train,
        "df_train": df_train,
    }

    if not df_pred.empty:
        X_raw_pred = torch.from_numpy(df_pred[feature_cols].values.astype(np.float32))
        intercept_pred = torch.ones(X_raw_pred.size(0), 1)
        X_pred_full = torch.cat([X_raw_pred, intercept_pred], dim=1)
        coords_pred_model = torch.from_numpy(coords_pred_model_np.astype(np.float32))
        coords_pred_raw_t = torch.from_numpy(coords_pred_raw.astype(np.float32))

        if use_fault:
            theta_pred = torch.from_numpy(angle_pred.astype(np.float32))
        else:
            theta_pred = torch.full((coords_pred_model.size(0),), float(theta_train[0]), dtype=torch.float32)
    else:
        X_pred_full = torch.empty((0, X_train_full.size(1)))
        coords_pred_model = torch.empty((0, 2))
        coords_pred_raw_t = torch.empty((0, 2))
        theta_pred = torch.empty((0,))

    predict_bundle = {
        "coords_pred_model": coords_pred_model,
        "coords_pred_raw": coords_pred_raw_t,
        "X_pred": X_pred_full,
        "theta_pred": theta_pred,
        "df_pred": df_pred,
    }

    meta = {
        "feature_cols": feature_cols,
        "coord_cols": coord_cols,
        "coord_center": coord_center,
        "coord_scale": coord_scale,
        "use_fault": use_fault,
        "fault_density_radius": fault_density_radius,
    }
    return train_bundle, predict_bundle, meta


# =========================
# 5. 训练与预测
# =========================

def train_model(
    train_bundle,
    H=120,
    W=100,
    epochs=10,
    train_batch_size=16,
    lr=1e-4,
    weight_decay=1e-5,
    parallel_scale=3.0,
    perpendicular_scale=1.0,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coords = train_bundle["coords_train_model"].to(device)
    X = train_bundle["X_train"].to(device)
    y = train_bundle["y_train"].to(device)
    theta = train_bundle["theta_train"].to(device)

    valid = (~torch.isnan(X).any(dim=1)) & (~torch.isnan(y).squeeze()) & (~torch.isnan(coords).any(dim=1)) & (~torch.isnan(theta))
    coords, X, y, theta = coords[valid], X[valid], y[valid], theta[valid]

    beta_ols = compute_ols(X, y)
    model = ACAWLR(x_dim=X.size(1), beta_ols=beta_ols).to(device)

    grid_u, grid_v = make_grid_from_coords(coords, H=H, W=W, device=device)

    dataset = TensorDataset(coords, X, y, theta)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    print("开始训练...")
    print(f"device={device}, samples={len(dataset)}, batch_size={train_batch_size}, grid={H}x{W}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0

        for coords_b, X_b, y_b, theta_b in dataloader:
            optimizer.zero_grad(set_to_none=True)

            aniso = generate_fault_anisotropic_grid_batch(
                grid_u, grid_v, coords_b, theta_b,
                parallel_scale=parallel_scale,
                perpendicular_scale=perpendicular_scale,
            )
            pred = model(aniso, X_b)
            loss = criterion(pred, y_b)

            if torch.isnan(loss) or torch.isinf(loss):
                print("警告：当前 batch 出现 NaN/Inf，已跳过。")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = coords_b.size(0)
            epoch_loss += loss.item() * batch_n
            seen += batch_n

        avg_loss = epoch_loss / max(seen, 1)
        print(f"Epoch [{epoch + 1:02d}/{epochs}] | Average Loss: {avg_loss:.6f}")

    print("训练结束。")
    return model, (grid_u, grid_v)


@torch.no_grad()
def predict_model(
    model,
    predict_bundle,
    grid_tuple,
    batch_size=16,
    parallel_scale=3.0,
    perpendicular_scale=1.0,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    coords = predict_bundle["coords_pred_model"].to(device)
    X = predict_bundle["X_pred"].to(device)
    theta = predict_bundle["theta_pred"].to(device)

    if coords.numel() == 0:
        return np.array([])

    grid_u, grid_v = grid_tuple
    dataset = TensorDataset(coords, X, theta)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs = []
    model.eval()
    for coords_b, X_b, theta_b in dataloader:
        aniso = generate_fault_anisotropic_grid_batch(
            grid_u, grid_v, coords_b, theta_b,
            parallel_scale=parallel_scale,
            perpendicular_scale=perpendicular_scale,
        )
        logits = model(aniso, X_b)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))

    return np.concatenate(probs, axis=0)


# =========================
# 6. 主程序配置
# =========================

if __name__ == "__main__":
    # 改成你的真实路径。Windows 路径前面建议加 r，避免反斜杠转义。
    DATA_CSV_PATH = r"D:\code\ResearchPractice\data\geochemistry\沉积物地球化学数据.csv"

    # 断裂带文件路径。
    # 支持：
    #   1) CSV: x1,y1,x2,y2,weight 可选
    #   2) shp/geojson/gpkg: 需要安装 geopandas
    # 没有就先填 None，代码会退回到矿点 PCA 方向。
    FAULT_PATH = None
    # 例子：FAULT_PATH = r"D:\code\ResearchPractice\fault_segments.csv"
    # 例子：FAULT_PATH = r"D:\code\ResearchPractice\faults.shp"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_bundle, predict_bundle, meta = data_procession(
        DATA_CSV_PATH,
        fault_path=FAULT_PATH,
        cu_col="Cu_ppm",
        coord_cols=("x", "y"),
        base_feature_cols=("Au_ppm", "Mo_ppm", "Fe_pct"),
        cu_threshold=3000,
        fault_density_radius=None,
    )

    print("实际进入模型的特征：", meta["feature_cols"])
    print("是否使用 fault：", meta["use_fault"])

    model, grid_tuple = train_model(
        train_bundle,
        H=120,
        W=100,
        epochs=10,
        train_batch_size=16,
        lr=1e-4,
        weight_decay=1e-5,
        parallel_scale=3.0,
        perpendicular_scale=1.0,
        device=device,
    )

    # 如果有 Cu_ppm 缺失的待预测点，则输出预测概率。
    probs = predict_model(
        model,
        predict_bundle,
        grid_tuple,
        batch_size=16,
        parallel_scale=3.0,
        perpendicular_scale=1.0,
        device=device,
    )

    if len(probs) > 0:
        out_df = predict_bundle["df_pred"].copy()
        out_df["prospectivity_probability"] = probs
        out_path = Path("ACAWLR_prediction_result.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"预测结果已保存：{out_path.resolve()}")
    else:
        print("没有待预测样本，未输出预测文件。")

