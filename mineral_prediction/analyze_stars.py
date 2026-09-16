from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import correlate, maximum_filter


IMAGE = Path(r"D:\code\ResearchPractice\mineral_prediction\tmp\pdf_pages\figure2G.png")
OUT = IMAGE.parent


def score_star(rgb, center, fill_rgb):
    cx, cy = center
    template = rgb[cy - 10 : cy + 11, cx - 10 : cx + 11]
    gray_t = template.mean(axis=2)
    black_mask = gray_t < 90
    target = np.asarray(fill_rgb, dtype=float)
    fill_mask = np.linalg.norm(template.astype(float) - target, axis=2) < 55

    gray = rgb.mean(axis=2)
    blackness = np.clip((120.0 - gray) / 120.0, 0, 1)
    color_dist = np.linalg.norm(rgb.astype(float) - target, axis=2)
    color_similarity = np.exp(-(color_dist**2) / (2 * 55.0**2))

    black_score = correlate(blackness, black_mask.astype(float), mode="constant") / black_mask.sum()
    fill_score = correlate(color_similarity, fill_mask.astype(float), mode="constant") / fill_mask.sum()
    shape_score = black_score - 0.35 * correlate(blackness, fill_mask.astype(float), mode="constant") / fill_mask.sum()
    return 0.72 * black_score + 0.28 * fill_score, shape_score


def peaks(score, threshold):
    allowed = np.zeros_like(score, dtype=bool)
    allowed[100:725, 40:790] = True
    local = score == maximum_filter(score, size=13)
    ys, xs = np.where(local & allowed & (score >= threshold))
    order = np.argsort(score[ys, xs])[::-1]
    return [(int(xs[i]), int(ys[i]), float(score[ys[i], xs[i]])) for i in order]


def main():
    image = Image.open(IMAGE).convert("RGB")
    rgb = np.asarray(image)
    train_score, train_shape = score_star(rgb, (11, 46), (250, 163, 27))
    valid_score, valid_shape = score_star(rgb, (11, 77), (129, 211, 243))
    train = peaks(train_score, 0.35)
    valid = peaks(valid_score, 0.34)

    print("TRAIN")
    for row in train:
        print(row)
    print("VALID")
    for row in valid:
        print(row)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for x, y, s in train:
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 0, 255), width=2)
        draw.text((x + 8, y - 8), f"{s:.2f}", fill=(255, 0, 255))
    for x, y, s in valid:
        draw.rectangle((x - 8, y - 8, x + 8, y + 8), outline=(0, 255, 0), width=2)
        draw.text((x + 8, y + 2), f"{s:.2f}", fill=(0, 130, 0))
    overlay.save(OUT / "figure2G_star_candidates.png")

    shape = np.maximum(train_shape, valid_shape)
    all_stars = peaks(shape, 0.18)[:100]
    shape_overlay = image.copy()
    draw = ImageDraw.Draw(shape_overlay)
    for i, (x, y, s) in enumerate(all_stars, 1):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 0, 255), width=2)
        draw.text((x + 7, y - 8), f"{i}:{s:.2f}", fill=(255, 0, 255))
    shape_overlay.save(OUT / "figure2G_all_star_candidates.png")
    print("ALL")
    for i, row in enumerate(all_stars, 1):
        print(i, row)


if __name__ == "__main__":
    main()
