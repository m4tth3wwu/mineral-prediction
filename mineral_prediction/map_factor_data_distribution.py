from __future__ import annotations

import gzip
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm, TwoSlopeNorm
from pyproj import Transformer
from shapely.geometry import box


PROJECT = Path(r"D:\code\ResearchPractice")
WORKSPACE = Path(__file__).resolve().parent
OUTPUT = WORKSPACE / "output" / "factor_distribution_maps"
OUTPUT.mkdir(parents=True, exist_ok=True)

GEOCHEM = PROJECT / "data/geochemistry/沉积物地球化学数据.csv"
FAULTS = PROJECT / "data/faults/GeologyFaults_USCanada.shp"
GRAVITY = PROJECT / "data/gravity" / "GRIDS" / "grid_xyz" / "bouguer.xyz.gz"
POINT_DIR = WORKSPACE / "output" / "paper_point_matching"

WEST, EAST, SOUTH, NORTH = -125.0, -102.0, 30.0, 50.0
STEP = 0.5
LON_EDGES = np.arange(WEST, EAST + STEP, STEP)
LAT_EDGES = np.arange(SOUTH, NORTH + STEP, STEP)
FEATURES = ["Cu_ppm", "Au_ppm", "Mo_ppm", "Fe_pct"]
FEATURE_LABELS = {
    "Cu_ppm": "Cu高值样本密度",
    "Au_ppm": "Au高值样本密度",
    "Mo_ppm": "Mo高值样本密度",
    "Fe_pct": "Fe高值样本密度",
}

STATE_LABELS = {
    "WA": (-120.7, 47.4), "OR": (-120.7, 44.1), "CA": (-120.0, 36.7),
    "NV": (-116.7, 39.1), "ID": (-114.5, 44.5), "MT": (-109.8, 47.1),
    "WY": (-107.5, 43.0), "UT": (-111.5, 39.2), "AZ": (-111.8, 34.3),
    "NM": (-105.9, 34.5), "CO": (-105.8, 39.0), "BC": (-122.1, 49.45),
    "AB": (-114.0, 49.45),
}

REGION_CENTERS = {
    "华盛顿州": (-120.7, 47.4), "俄勒冈州": (-120.7, 44.1),
    "加利福尼亚州": (-119.7, 36.7), "内华达州": (-116.7, 39.1),
    "爱达荷州": (-114.5, 44.5), "蒙大拿州": (-109.8, 47.1),
    "怀俄明州": (-107.5, 43.0), "犹他州": (-111.5, 39.2),
    "亚利桑那州": (-111.8, 34.3), "新墨西哥州": (-105.9, 34.5),
    "科罗拉多州": (-105.8, 39.0), "不列颠哥伦比亚省南部": (-122.0, 49.5),
    "阿尔伯塔省南部": (-114.0, 49.5),
}


def configure_fonts():
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_geochemistry():
    usecols = ["longitude", "latitude", *FEATURES]
    data = pd.read_csv(GEOCHEM, usecols=usecols, low_memory=False)
    for col in usecols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["longitude", "latitude"])
    return data[
        data["longitude"].between(WEST, EAST)
        & data["latitude"].between(SOUTH, NORTH)
    ].reset_index(drop=True)


def load_faults():
    faults = gpd.read_file(FAULTS)
    if faults.crs is None:
        faults = faults.set_crs(4326)
    faults = faults.to_crs(4326)
    return gpd.clip(faults, gpd.GeoDataFrame(geometry=[box(WEST, SOUTH, EAST, NORTH)], crs=4326))


def load_gravity():
    gravity = pd.read_csv(
        GRAVITY,
        sep=r"\s+",
        names=["longitude", "latitude", "gravity_bouguer"],
        compression="gzip",
        dtype=np.float32,
    ).dropna()
    return gravity[
        gravity["longitude"].between(WEST, EAST)
        & gravity["latitude"].between(SOUTH, NORTH)
    ].reset_index(drop=True)


def load_deposits():
    train = pd.read_csv(POINT_DIR / "paper_figure2g_training_44_model_ready.csv")
    validation = pd.read_csv(POINT_DIR / "paper_figure2g_validation_16_model_ready.csv")
    for frame in (train, validation):
        frame.columns = [str(c).lower() for c in frame.columns]
    return train, validation


def histogram(lon, lat, weights=None):
    values, _, _ = np.histogram2d(lat, lon, bins=[LAT_EDGES, LON_EDGES], weights=weights)
    return values


def region_name(lon, lat):
    # Use broad state/province envelopes first, then nearest regional center.
    if lat >= 49.0:
        return "不列颠哥伦比亚省南部" if lon < -120.0 else "阿尔伯塔省南部"
    if 45.5 <= lat < 49.0 and lon <= -116.8:
        return "华盛顿州"
    if 42.0 <= lat < 46.0 and lon <= -116.4:
        return "俄勒冈州"
    if 45.0 <= lat < 49.0 and -116.0 <= lon <= -102.0:
        return "蒙大拿州"
    if 42.0 <= lat < 49.0 and -117.3 <= lon <= -111.0:
        return "爱达荷州"
    if 41.0 <= lat < 45.0 and -111.2 <= lon <= -104.0:
        return "怀俄明州"
    if 37.0 <= lat < 42.0 and -114.2 <= lon <= -109.0:
        return "犹他州"
    if 37.0 <= lat < 41.0 and -109.0 <= lon <= -102.0:
        return "科罗拉多州"
    if 35.0 <= lat < 42.0 and -120.0 <= lon <= -114.0:
        return "内华达州"
    if 31.0 <= lat < 37.0 and -115.0 <= lon <= -109.0:
        return "亚利桑那州"
    if 31.0 <= lat < 37.0 and -109.0 <= lon <= -102.0:
        return "新墨西哥州"
    if 32.0 <= lat < 42.0 and lon < -114.0:
        return "加利福尼亚州"
    best = min(
        REGION_CENTERS,
        key=lambda name: ((lon - REGION_CENTERS[name][0]) * math.cos(math.radians(lat))) ** 2
        + (lat - REGION_CENTERS[name][1]) ** 2,
    )
    return best


def select_hotspots(grid, count=3, min_separation_deg=1.5):
    candidates = []
    for iy, ix in np.dstack(np.unravel_index(np.argsort(grid.ravel())[::-1], grid.shape))[0]:
        value = float(grid[iy, ix])
        if value <= 0:
            break
        lon = (LON_EDGES[ix] + LON_EDGES[ix + 1]) / 2
        lat = (LAT_EDGES[iy] + LAT_EDGES[iy + 1]) / 2
        if all(math.hypot(lon - p[0], lat - p[1]) >= min_separation_deg for p in candidates):
            candidates.append((lon, lat, value))
        if len(candidates) >= count:
            break
    return candidates


def sample_fault_lines(faults, interval_m=20_000.0):
    metric = faults.to_crs("ESRI:102008")
    xs, ys = [], []
    for geom in metric.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            if part.length <= 0:
                continue
            distances = np.arange(0.0, part.length + 1.0, interval_m)
            for distance in distances:
                pt = part.interpolate(float(distance))
                xs.append(pt.x)
                ys.append(pt.y)
    transformer = Transformer.from_crs("ESRI:102008", 4326, always_xy=True)
    lon, lat = transformer.transform(np.asarray(xs), np.asarray(ys))
    return np.asarray(lon), np.asarray(lat), interval_m / 1000.0


def draw_state_labels(ax):
    for abbrev, (lon, lat) in STATE_LABELS.items():
        ax.text(lon, lat, abbrev, ha="center", va="center", fontsize=6.2,
                color="#4b5563", alpha=0.75, fontweight="bold", zorder=6)


def base_axis(ax, faults, title):
    faults.plot(ax=ax, color="#555b63", linewidth=0.23, alpha=0.32, zorder=3)
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)
    ax.set_title(title, fontsize=11.2, fontweight="bold", pad=6)
    ax.set_xticks(np.arange(-125, -101, 5))
    ax.set_yticks(np.arange(30, 51, 5))
    ax.grid(color="#aeb6bd", linewidth=0.35, alpha=0.45)
    ax.set_xlabel("经度", fontsize=8)
    ax.set_ylabel("纬度", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#f7f8f9")
    draw_state_labels(ax)


def plot_log_density(ax, grid, cmap="YlOrRd", label="每0.5°网格点数"):
    masked = np.ma.masked_where(grid <= 0, grid)
    vmax = max(float(np.nanmax(grid)), 1.0)
    mesh = ax.pcolormesh(LON_EDGES, LAT_EDGES, masked, shading="auto", cmap=cmap,
                         norm=LogNorm(vmin=1, vmax=vmax), alpha=0.82, zorder=2)
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.76, pad=0.018)
    cbar.set_label(label, fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    return mesh


def annotate_hotspots(ax, hotspots):
    region_counts = {}
    for rank, (lon, lat, _) in enumerate(hotspots, start=1):
        region = region_name(lon, lat)
        region_counts[region] = region_counts.get(region, 0) + 1
        suffix = "" if region_counts[region] == 1 else f"-{region_counts[region]}"
        ax.scatter(lon, lat, s=28, marker="o", facecolor="white", edgecolor="#9b1c1c", linewidth=1.1, zorder=8)
        ax.annotate(f"{rank}. {region}{suffix}", (lon, lat), xytext=(4, 5), textcoords="offset points",
                    fontsize=6.4, color="#7f1d1d", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.78), zorder=9)


def build_maps():
    configure_fonts()
    geochem = load_geochemistry()
    faults = load_faults()
    gravity = load_gravity()
    train_deposits, validation_deposits = load_deposits()

    valid_any = geochem[FEATURES].notna().any(axis=1)
    geochem_grid = histogram(geochem.loc[valid_any, "longitude"], geochem.loc[valid_any, "latitude"])

    high_grids = {}
    thresholds = {}
    stats_rows = []
    for feature in FEATURES:
        valid = geochem[["longitude", "latitude", feature]].dropna()
        positive = valid[valid[feature] > 0]
        threshold = float(positive[feature].quantile(0.90)) if len(positive) else float("nan")
        high = positive[positive[feature] >= threshold]
        grid = histogram(high["longitude"], high["latitude"])
        high_grids[feature] = grid
        thresholds[feature] = threshold
        for rank, (lon, lat, value) in enumerate(select_hotspots(grid, count=4), start=1):
            stats_rows.append({
                "factor": feature,
                "definition": "positive-value 90th percentile and above",
                "threshold": threshold,
                "rank": rank,
                "region": region_name(lon, lat),
                "longitude": lon,
                "latitude": lat,
                "count_in_0.5deg_cell": int(value),
                "valid_observations": len(valid),
                "high_value_observations": len(high),
            })

    fault_lon, fault_lat, interval_km = sample_fault_lines(faults)
    fault_grid = histogram(fault_lon, fault_lat, weights=np.full(len(fault_lon), interval_km))
    for rank, (lon, lat, value) in enumerate(select_hotspots(fault_grid, count=4), start=1):
        stats_rows.append({
            "factor": "fault_length_density",
            "definition": "approximate fault length sampled every 20 km",
            "threshold": np.nan,
            "rank": rank,
            "region": region_name(lon, lat),
            "longitude": lon,
            "latitude": lat,
            "count_in_0.5deg_cell": value,
            "valid_observations": len(fault_lon),
            "high_value_observations": np.nan,
        })

    gravity_count = histogram(gravity["longitude"], gravity["latitude"])
    gravity_sum = histogram(gravity["longitude"], gravity["latitude"], weights=gravity["gravity_bouguer"])
    gravity_mean = np.divide(gravity_sum, gravity_count, out=np.full_like(gravity_sum, np.nan), where=gravity_count > 0)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10.7), constrained_layout=True)
    axes = axes.ravel()

    base_axis(axes[0], faults, "A. 地球化学有效采样密度")
    plot_log_density(axes[0], geochem_grid, cmap="YlGnBu", label="有效采样点数/0.5°网格")
    annotate_hotspots(axes[0], select_hotspots(geochem_grid, count=4))
    axes[0].scatter(train_deposits["longitude"], train_deposits["latitude"], s=11, c="#00a6a6",
                    marker="o", edgecolor="white", linewidth=0.35, label="44训练矿床", zorder=10)
    axes[0].scatter(validation_deposits["longitude"], validation_deposits["latitude"], s=28, c="#b12a90",
                    marker="*", edgecolor="white", linewidth=0.35, label="16验证矿床", zorder=10)
    axes[0].legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.85)

    for ax, feature, letter in zip(axes[1:5], FEATURES, ["B", "C", "D", "E"]):
        base_axis(ax, faults, f"{letter}. {FEATURE_LABELS[feature]}（≥P90={thresholds[feature]:.3g}）")
        plot_log_density(ax, high_grids[feature], cmap="YlOrRd", label="高值样本数/0.5°网格")
        annotate_hotspots(ax, select_hotspots(high_grids[feature], count=3))

    base_axis(axes[5], faults, "F. 断层长度密度（约20 km步长）")
    plot_log_density(axes[5], fault_grid, cmap="magma_r", label="近似断层长度(km)/0.5°网格")
    annotate_hotspots(axes[5], select_hotspots(fault_grid, count=3))

    base_axis(axes[6], faults, "G. Bouguer重力异常空间分布")
    finite = gravity_mean[np.isfinite(gravity_mean)]
    q02, q98 = np.nanquantile(finite, [0.02, 0.98])
    center = 0.0 if q02 < 0 < q98 else float(np.nanmedian(finite))
    mesh = axes[6].pcolormesh(LON_EDGES, LAT_EDGES, np.ma.masked_invalid(gravity_mean), shading="auto",
                              cmap="RdBu_r", norm=TwoSlopeNorm(vmin=q02, vcenter=center, vmax=q98),
                              alpha=0.84, zorder=2)
    cbar = plt.colorbar(mesh, ax=axes[6], shrink=0.76, pad=0.018)
    cbar.set_label("Bouguer重力异常（原始数据单位）", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    base_axis(axes[7], faults, "H. 三类数据覆盖总览")
    gravity_mask = np.where(gravity_count > 0, 1.0, np.nan)
    axes[7].pcolormesh(LON_EDGES, LAT_EDGES, gravity_mask, shading="auto",
                       cmap=ListedColormap(["#60a5fa"]), alpha=0.24, zorder=1)
    geo_mask = np.where(geochem_grid > 0, 1.0, np.nan)
    axes[7].pcolormesh(LON_EDGES, LAT_EDGES, geo_mask, shading="auto",
                       cmap=ListedColormap(["#f59e0b"]), alpha=0.32, zorder=2)
    gravity_outline = (gravity_count > 0).astype(float)
    x_centers = (LON_EDGES[:-1] + LON_EDGES[1:]) / 2
    y_centers = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2
    if np.any(gravity_outline):
        axes[7].contour(x_centers, y_centers, gravity_outline, levels=[0.5], colors="#1463a5", linewidths=1.2, zorder=5)
    axes[7].text(WEST + 0.6, SOUTH + 1.8, "橙色：地球化学有数据", fontsize=7.2,
                 bbox=dict(fc="white", ec="none", alpha=0.85), zorder=8)
    axes[7].text(WEST + 0.6, SOUTH + 0.8, "蓝线内：Bouguer重力有数据", color="#1463a5", fontsize=7.2,
                 bbox=dict(fc="white", ec="none", alpha=0.85), zorder=8)
    axes[7].text(WEST + 0.6, SOUTH + 2.8, "灰线：断层数据", color="#4b5563", fontsize=7.2,
                 bbox=dict(fc="white", ec="none", alpha=0.85), zorder=8)

    fig.suptitle("西部北美成矿因子数据密度与空间分布总览", fontsize=17, fontweight="bold", color="#173b53")
    fig.text(0.5, 0.008,
             "说明：元素热点为研究区内正值样本的前10%（P90以上）在0.5°网格中的样本密度；颜色深浅同时受采样数量影响，不等同于连续地质储量。",
             ha="center", fontsize=9, color="#4b5563")
    map_path = OUTPUT / "成矿因子密度与空间分布总览.png"
    fig.savefig(map_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(OUTPUT / "成矿因子热点统计.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 成矿因子地图说明",
        "",
        "## 判读口径",
        "",
        f"- 研究范围：{WEST}°至{EAST}°，{SOUTH}°至{NORTH}°。",
        f"- 研究区内地球化学坐标点：{len(geochem):,}个；Bouguer重力点：{len(gravity):,}个。",
        "- 元素高值热点定义为各元素正值观测中的P90及以上，并统计在0.5°×0.5°网格中的样本数。",
        "- 断层密度用断层线每20 km采样一次后近似换算线长密度。",
        "- 深色既可能表示异常样本集中，也可能受采样密度影响；不能直接解释为矿体规模或储量。",
        "",
        "## 主要集中地区",
        "",
    ]
    for feature in FEATURES:
        subset = stats[stats["factor"].eq(feature)].sort_values("rank")
        regions = "、".join(dict.fromkeys(subset["region"].tolist()))
        lines.append(
            f"- **{feature}**：P90阈值为{thresholds[feature]:.4g}；高值样本密度较集中的地区为{regions}。"
        )
    fault_subset = stats[stats["factor"].eq("fault_length_density")].sort_values("rank")
    fault_regions = "、".join(dict.fromkeys(fault_subset["region"].tolist()))
    lines.extend([
        f"- **断层**：断层线长度密度较高的地区为{fault_regions}。",
        "- **Bouguer重力**：覆盖主要位于美国西部研究区；图G显示异常值的连续空间变化，图H蓝线显示其数据覆盖边界。",
        "",
        "## 与模型变量的关系",
        "",
        "Cu、Au、Mo、Fe原始采样点经12邻点IDW插值到1 km模型网格；断层距离、断层密度和方向集中度直接由断层线计算；Bouguer重力使用最近网格点匹配。因此本图用于判断数据覆盖与高值聚集位置，不是模型最终矿床概率图。",
    ])
    (OUTPUT / "成矿因子地图说明.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"geochemistry_points={len(geochem)} gravity_points={len(gravity)} fault_sample_points={len(fault_lon)}")
    print("thresholds=", thresholds)
    print(stats[["factor", "rank", "region", "longitude", "latitude", "count_in_0.5deg_cell"]].to_string(index=False))
    print(map_path)


if __name__ == "__main__":
    build_maps()
