# Central Ghana Groundwater Benchmark

This folder contains the publication-ready artefacts for the 103-borehole Central Region of Ghana study prepared for Computers and Geosciences.

## Contents
- `manuscript/` - final manuscript source in Elsevier `elsarticle` format
- `code/` - modelling and figure-generation scripts used for the package
- `data/` - borehole records, shapefile components, and GitHub-sized supporting datasets for the study area
- `results/` - final metrics, comparison tables, manifests, and report files
- `figures/` - publication figures generated for the manuscript
- `models/` - saved trained model artefacts
- `docs/` - reproducibility and metadata files

## Final headline results
- Stage 1 Accuracy: 0.510
- Stage 1 F1: 0.600
- Stage 2 R2: 0.025
- Stage 2 RMSE: 1.279
- Stage 2 MAE: 0.940
- Stage 2 Spearman rho: 0.052
- Stage 2 Zone Accuracy: 0.368
- Stage 2 Direction Accuracy: 0.509
- Stage 2 Bias: -0.068 m3/h

## SSL ablation summary
The tuned self-supervised latent-feature ablation is included for transparency.
- Stage 2 R2: 0.040
- Stage 2 RMSE: 1.270
- Stage 2 MAE: 0.918
- Stage 2 Spearman rho: 0.139
- Stage 2 Zone Accuracy: 0.386
- Stage 2 Direction Accuracy: 0.456
- Stage 2 Bias: -0.077 m3/h

## Groundwater potential map scale
The groundwater potential map uses four classes:
- 0.0-0.5: Very low
- 0.5-1.5: Moderate
- 1.5-2.0: High
- 2.0-2.5 and above: Very high

## Notes
- The paper uses district-disjoint GroupKFold validation.
- The package keeps the main baseline as the primary result and the SSL run as a sensitivity analysis.
- British English spelling and publication-style prose are used throughout the manuscript.
- The raw borehole table `Boreholes.csv` is intentionally omitted from the GitHub snapshot.
- Two oversized raw support tables, `SMAP_SOIL_MOISTURE.csv` and `MODIS_LAND_COVER_GHANA.csv`, are intentionally omitted from the GitHub snapshot because of GitHub file-size constraints.
- The repository keeps the lighter GitHub-sized package for dissemination; excluded bulky local files remain outside the tracked snapshot.
