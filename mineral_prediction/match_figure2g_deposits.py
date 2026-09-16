import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.signal import correlate2d
from sklearn.neighbors import BallTree


ROOT = Path(r"D:\code\ResearchPractice\mineral_prediction")
DEFAULT_FIGURE = ROOT / "tmp" / "pdf_pages" / "figure2G_original.jpg"
DEFAULT_DEPOSITS = ROOT / "data" / "usgs_porpyhry" / "western_us_deposit_points_used.csv"
DEFAULT_OUTPUT = ROOT / "output" / "paper_point_matching"

# Calibrated from Figure 2G tick marks:
# x = 23.16 * longitude + 2943.2
# y = -23.2 * latitude + 1229.0
X_SLOPE = 23.16
X_INTERCEPT = 2943.2
Y_SLOPE = -23.2
Y_INTERCEPT = 1229.0


def pixel_to_lonlat(x, y):
    return (x - X_INTERCEPT) / X_SLOPE, (y - Y_INTERCEPT) / Y_SLOPE


def lonlat_to_pixel(lon, lat):
    return X_SLOPE * lon + X_INTERCEPT, Y_SLOPE * lat + Y_INTERCEPT


def normalized_correlation(image_feature, center_x, center_y, radius=6):
    template = image_feature[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ].copy()
    template -= template.mean()
    numerator = correlate2d(image_feature, template, mode="same")
    n = template.size
    local_sum = uniform_filter(image_feature, size=template.shape, mode="constant") * n
    local_sum_sq = uniform_filter(image_feature**2, size=template.shape, mode="constant") * n
    local_variance_sum = np.maximum(local_sum_sq - local_sum**2 / n, 1e-9)
    denominator = np.sqrt(local_variance_sum * np.sum(template**2))
    return numerator / denominator


def detect_paper_stars(figure_path: Path) -> pd.DataFrame:
    image = np.asarray(Image.open(figure_path).convert("RGB"), dtype=float)
    if image.shape[:2] != (588, 613):
        raise ValueError(
            f"Expected the extracted Figure 2G image to be 613 x 588 pixels; got {image.shape[1]} x {image.shape[0]}."
        )

    blackness = 1.0 - image.mean(axis=2) / 255.0
    # Clean isolated examples from the figure itself: one training star and one validation star.
    training_ncc = normalized_correlation(blackness, 507, 274)
    validation_ncc = normalized_correlation(blackness, 277, 405)
    score = np.maximum(training_ncc, validation_ncc)

    study_mask = np.zeros_like(score, dtype=bool)
    study_mask[85:535, 48:580] = True
    local_peaks = (score == maximum_filter(score, size=8)) & study_mask & (score > 0.40)
    ys, xs = np.where(local_peaks)

    rows = []
    for x, y in zip(xs, ys):
        patch = image[y - 6 : y + 7, x - 6 : x + 7]
        center = image[y - 3 : y + 4, x - 3 : x + 4]
        dark_count = int((patch.max(axis=2) < 105).sum())
        gold_count = int(
            (
                (center[:, :, 0] - center[:, :, 2] > 35)
                & (center[:, :, 0] > 140)
                & (center[:, :, 1] > 75)
            ).sum()
        )
        cyan_count = int(
            (
                (center[:, :, 2] - center[:, :, 0] > 30)
                & (center[:, :, 2] > 130)
                & (center[:, :, 1] > 110)
            ).sum()
        )
        if dark_count < 25 or max(gold_count, cyan_count) < 10:
            continue

        lon, lat = pixel_to_lonlat(x, y)
        rows.append(
            {
                "paper_pixel_x": int(x),
                "paper_pixel_y": int(y),
                "paper_longitude_approx": float(lon),
                "paper_latitude_approx": float(lat),
                "paper_split": "validation" if cyan_count > gold_count else "training",
                "detection_score": float(score[y, x]),
                "dark_pixel_count": dark_count,
                "gold_fill_count": gold_count,
                "cyan_fill_count": cyan_count,
            }
        )

    stars = pd.DataFrame(rows).sort_values(["paper_pixel_y", "paper_pixel_x"]).reset_index(drop=True)
    stars.insert(0, "paper_star_id", np.arange(1, len(stars) + 1))
    if len(stars) != 60:
        raise ValueError(f"Expected 60 paper stars after calibrated detection; detected {len(stars)}.")
    return stars


def match_usgs_points(stars: pd.DataFrame, deposit_path: Path, max_match_km: float):
    deposits = pd.read_csv(deposit_path)
    required = {"deposit_latitude", "deposit_longitude", "DEPOSIT"}
    missing = required - set(deposits.columns)
    if missing:
        raise ValueError(f"USGS candidate file is missing columns: {sorted(missing)}")

    tree = BallTree(
        np.radians(deposits[["deposit_latitude", "deposit_longitude"]].to_numpy(float)),
        metric="haversine",
    )
    star_coordinates = stars[["paper_latitude_approx", "paper_longitude_approx"]].to_numpy(float)
    distances_rad, indices = tree.query(np.radians(star_coordinates), k=3)
    distances_km = distances_rad * 6371.0088

    detail_rows = []
    model_rows = []
    unmatched_rows = []
    used_usgs_indices = set()
    for i, star in stars.iterrows():
        nearest_index = int(indices[i, 0])
        nearest_distance = float(distances_km[i, 0])
        is_match = nearest_distance <= max_match_km and nearest_index not in used_usgs_indices
        nearest = deposits.iloc[nearest_index]
        row = star.to_dict()
        row.update(
            {
                "match_status": "matched" if is_match else "unmatched",
                "match_confidence": (
                    "high" if is_match and nearest_distance <= 10 else "medium" if is_match else "unmatched"
                ),
                "match_distance_km": nearest_distance,
                "nearest_usgs_name": nearest["DEPOSIT"],
                "nearest_usgs_longitude": float(nearest["deposit_longitude"]),
                "nearest_usgs_latitude": float(nearest["deposit_latitude"]),
                "second_usgs_name": deposits.iloc[int(indices[i, 1])]["DEPOSIT"],
                "second_distance_km": float(distances_km[i, 1]),
                "third_usgs_name": deposits.iloc[int(indices[i, 2])]["DEPOSIT"],
                "third_distance_km": float(distances_km[i, 2]),
            }
        )
        detail_rows.append(row)

        if is_match:
            used_usgs_indices.add(nearest_index)
            model_row = nearest.to_dict()
            model_row.pop("deposit_longitude", None)
            model_row.pop("deposit_latitude", None)
            model_row.update(
                {
                    "LONGITUDE": float(nearest["deposit_longitude"]),
                    "LATITUDE": float(nearest["deposit_latitude"]),
                    "COUNTRY": nearest.get("COUNTRY", "United States"),
                    "paper_star_id": int(star["paper_star_id"]),
                    "paper_split": star["paper_split"],
                    "paper_match_distance_km": nearest_distance,
                }
            )
            model_rows.append(model_row)
        else:
            unmatched_rows.append(row)

    details = pd.DataFrame(detail_rows)
    model_ready = pd.DataFrame(model_rows)
    unmatched = pd.DataFrame(unmatched_rows)
    return details, model_ready, unmatched


def make_verification_overlay(figure_path, details, output_path):
    image = Image.open(figure_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for _, row in details.iterrows():
        x = int(row["paper_pixel_x"])
        y = int(row["paper_pixel_y"])
        color = (0, 210, 255) if row["paper_split"] == "validation" else (255, 0, 210)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=2)
        if row["match_status"] == "matched":
            mx, my = lonlat_to_pixel(row["nearest_usgs_longitude"], row["nearest_usgs_latitude"])
            draw.line((x, y, mx, my), fill=(20, 220, 40), width=1)
            draw.ellipse((mx - 2, my - 2, mx + 2, my + 2), fill=(20, 220, 40))
        else:
            draw.line((x - 7, y - 7, x + 7, y + 7), fill=(255, 0, 0), width=2)
            draw.line((x - 7, y + 7, x + 7, y - 7), fill=(255, 0, 0), width=2)
            draw.text((x + 8, y - 8), "unmatched", fill=(255, 0, 0))
    image.resize((1226, 1176), Image.Resampling.LANCZOS).save(output_path)


def make_geographic_match_map(details, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)
    for split, color, label in (
        ("training", "#f4a261", "Paper training stars"),
        ("validation", "#00b4d8", "Paper validation stars"),
    ):
        subset = details[(details["paper_split"] == split) & (details["match_status"] == "matched")]
        ax.scatter(
            subset["nearest_usgs_longitude"],
            subset["nearest_usgs_latitude"],
            marker="*",
            s=82,
            c=color,
            edgecolors="#202020",
            linewidths=0.4,
            label=label,
        )
    unmatched = details[details["match_status"] == "unmatched"]
    if len(unmatched):
        ax.scatter(
            unmatched["paper_longitude_approx"],
            unmatched["paper_latitude_approx"],
            marker="x",
            s=65,
            c="#d00000",
            linewidths=1.5,
            label="Paper star without USGS match",
        )
    ax.set_xlim(-125, -102)
    ax.set_ylim(30, 50)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Figure 2G stars matched to USGS porphyry occurrences")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Match Figure 2G star symbols to USGS porphyry occurrences.")
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--deposits", type=Path, default=DEFAULT_DEPOSITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-match-km", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stars = detect_paper_stars(args.figure)
    details, model_ready, unmatched = match_usgs_points(stars, args.deposits, args.max_match_km)

    details.to_csv(args.output_dir / "paper_figure2g_star_matches.csv", index=False, encoding="utf-8-sig")
    model_ready.to_csv(
        args.output_dir / "paper_figure2g_matched_deposits_model_ready.csv", index=False, encoding="utf-8-sig"
    )
    approximate_rows = []
    for _, row in unmatched.iterrows():
        approximate_rows.append(
            {
                "DEPOSIT": f"Figure2G_unmatched_star_{int(row['paper_star_id']):02d}",
                "COUNTRY": "United States",
                "LONGITUDE": float(row["paper_longitude_approx"]),
                "LATITUDE": float(row["paper_latitude_approx"]),
                "paper_star_id": int(row["paper_star_id"]),
                "paper_split": row["paper_split"],
                "paper_match_distance_km": np.nan,
                "coordinate_source": "Figure 2G georeferenced approximation",
            }
        )
    all_model_ready = pd.concat([model_ready, pd.DataFrame(approximate_rows)], ignore_index=True, sort=False)
    if "coordinate_source" not in all_model_ready.columns:
        all_model_ready["coordinate_source"] = "USGS matched occurrence"
    else:
        all_model_ready["coordinate_source"] = all_model_ready["coordinate_source"].fillna(
            "USGS matched occurrence"
        )
    all_model_ready = all_model_ready.sort_values("paper_star_id").reset_index(drop=True)
    all_model_ready.to_csv(
        args.output_dir / "paper_figure2g_all_60_points_model_ready.csv", index=False, encoding="utf-8-sig"
    )
    all_model_ready[all_model_ready["paper_split"].eq("training")].to_csv(
        args.output_dir / "paper_figure2g_training_44_model_ready.csv", index=False, encoding="utf-8-sig"
    )
    all_model_ready[all_model_ready["paper_split"].eq("validation")].to_csv(
        args.output_dir / "paper_figure2g_validation_16_model_ready.csv", index=False, encoding="utf-8-sig"
    )
    unmatched.to_csv(args.output_dir / "paper_figure2g_unmatched_stars.csv", index=False, encoding="utf-8-sig")
    make_verification_overlay(args.figure, details, args.output_dir / "paper_figure2g_match_overlay.png")
    make_geographic_match_map(details, args.output_dir / "paper_figure2g_matched_points_map.png")

    print(
        {
            "detected_stars": len(stars),
            "training_stars": int((stars["paper_split"] == "training").sum()),
            "validation_stars": int((stars["paper_split"] == "validation").sum()),
            "matched_usgs_points": len(model_ready),
            "unmatched_paper_stars": len(unmatched),
            "median_match_distance_km": float(details.loc[details.match_status == "matched", "match_distance_km"].median()),
        }
    )


if __name__ == "__main__":
    main()
