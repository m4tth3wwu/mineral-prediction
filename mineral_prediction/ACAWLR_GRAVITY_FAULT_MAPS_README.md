# ACAWLR with gravity, faults, and maps

The script `ACAWLR_gravity_fault_maps.py` uses:

- predictors: `Cu_ppm`, `Au_ppm`, `Mo_ppm`, `Fe_pct`;
- Bouguer gravity anomaly, gravity-match distance, and gravity availability;
- distance to nearest fault, local fault density, and local fault-direction confidence;
- local fault direction to rotate the anisotropic spatial proximity grid;
- USGS porphyry occurrence coordinates to create 2 km positive buffers.

## Full paper-scale run

```powershell
python .\ACAWLR_gravity_fault_maps.py
```

Defaults use five folds, 25 epochs, a `120 x 100` GASPG grid, a 20:1 negative-to-positive sampling cap, the Bouguer grid, and the fault shapefile under `D:\code\ResearchPractice`.

## Quick end-to-end check

```powershell
python .\ACAWLR_gravity_fault_maps.py `
  --output-dir .\output\acawlr_gravity_fault_maps_smoke `
  --neg-pos-ratio 1 --max-negatives 500 `
  --epochs 1 --batch-size 64 --grid-h 24 --grid-w 20
```

The quick check verifies execution and graphics only. Its metrics are not final scientific results.

## Main outputs

- `selected_training_samples_final.csv`: modeling records with geochemistry, gravity, and fault features.
- `cv_metrics.csv`: five-fold validation metrics.
- `cv_predictions.csv`: out-of-fold probability for each selected record.
- `deposit_prediction_probabilities.csv`: deposit-level probability estimated by inverse-distance weighting nearby out-of-fold predictions.
- `map_model_inputs.png`: Bouguer gravity, faults, samples, and deposits.
- `map_and_validation_dashboard.png`: deposit probabilities, regional probability map, ROC, and precision-recall curve.
- `model_fold_1.pt` through `model_fold_5.pt`: saved fold models.

## Interpretation warning

The deposit coordinates create the positive labels. Therefore, `deposit_prediction_probabilities.csv` is useful for visualization and ranking, but it is not an independent external validation. Independent validation requires holding out complete deposits or spatial regions before positive buffers are generated.
