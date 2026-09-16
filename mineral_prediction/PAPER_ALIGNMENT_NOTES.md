# ACAWLR paper-alignment notes

## What changed

- `Cu_ppm` is now a predictor together with `Au_ppm`, `Mo_ppm`, and `Fe_pct`.
- The binary target is proximity to a known porphyry copper occurrence, not a threshold applied to `Cu_ppm`.
- Known occurrences come from the USGS 2025 global porphyry copper database and are clipped to the paper map extent (`125°W–102°W`, `30°N–50°N`) and the United States.
- Samples within 2 km of an occurrence are positive, matching the buffer distance explicitly reported in the paper's supplemental methods. The distance is configurable with `--deposit-buffer-km`.
- The gravity input now defaults to the Bouguer grid named in the supplemental methods.
- GASPG defaults to `120 × 100`, as reported in Supplemental Table S1.
- Raw projected coordinates remain inputs to the spatial weighting network but are no longer included as ordinary covariates.

## Reproducibility limitation

The paper and supplement do not publish the coordinates assigned to the Figure 2G training and validation stars, nor the random split. The USGS points are therefore a traceable replacement label source, not a claim of exact recovery of the authors' unpublished split.

## Data source

Magnin, B. P., Graham, G. E., Huston, D. L., and Eglington, B. M. (2025), *A global database of porphyry copper deposits and prospects*, U.S. Geological Survey data release, https://doi.org/10.5066/P14CCESQ (CC0).

## Verified smoke result

With the default 2 km buffer, the cleaned study-area sediment data produced 713 positive and 14,260 sampled negative records. Positive samples span `Cu_ppm = 0–5000`, and 17 negative records have `Cu_ppm >= 3000`; this confirms the label is independent of the old copper-threshold rule.
