# Publication Reproducibility Guide

## Recommended command

Run the paper-compatible profile from the repository root:

```bash
python model.py --profile paper
```

For manuscript-method execution (two-stage framework with Spatial-Ridge stacking), run:

```bash
python model.py --profile manuscript
```

## Expected output files

After a successful run, the `results/` folder should contain:

- `enhanced_predictions.csv`
- `feature_importance.csv` (feature-level importance used in paper interpretation)
- `meta_model_weights.csv` (stacking model weights only)
- `research_report.txt` (paper summary with SBGI formula and metrics)
- `reproducibility_manifest.json` (profile, data sources, file checksums, metrics)
- `groundwater_potential_map.html`
- `groundwater_potential_map.png`
- `groundwater_potential.tif`
- `yield_heatmap.png`

## Reviewer-facing reproducibility evidence

Use `results/reproducibility_manifest.json` as the primary audit artifact:

- run profile (`paper` or `publication`)
- random seed
- exact input files used
- SHA256 checksums for key input files
- feature list used in training
- final metrics

## Notes

- `feature_importance.csv` contains actual feature importances.
- `meta_model_weights.csv` contains only stacker-level weights.
- SBGI is explicitly represented as `sbgi` and `sbgi_geology` in model features.
- The GitHub snapshot excludes `Boreholes.csv` from version control.
- The GitHub snapshot excludes `SMAP_SOIL_MOISTURE.csv` and `MODIS_LAND_COVER_GHANA.csv` because they exceed practical GitHub repository size limits.
- For full local reruns that require those raw tables, use the local archived copies or regenerate the inputs before executing the full pipeline.
