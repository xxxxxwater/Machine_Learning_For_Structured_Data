# Changelog

## Unreleased (development after v0.2.0)

- Added explicit `SDataFrame.as_sklearn()` data-only views for scikit-learn
  cross-validation, row/column selection, and stable feature names.
- Added indexed-target alignment checks and per-column dtype validation to
  prevent silent training/inference schema changes.
- Introduced `PurgedEventTimeSeriesSplit` to exclude past samples whose
  forward-looking labels overlap each validation fold.
- Made per-observation resampling atomic on failures.
- Added comprehensive CV, overlap, and copy-isolation regression tests.
- Extended CI with optional B-spline/LOWESS tests and Ruff correctness lint.
- Preserved the published v0.2.0 Release tag and GHCR image.

## 0.2.0 — 2026-10-09

A focused modernization of the historical 2017 research prototype.

### Fixed

- Package-wide import errors and legacy transformer syntax problems.
- Broken and inconsistent return contracts in `fit`, `transform`, and `fit_transform`.
- Incorrect row/feature orientation and feature dropping in Bag and Series extraction.
- Undefined variables in image and text feature extraction.
- Stale array/int API usage and silent validation failures.
- Broken `SDataFrame.join`, `SData.C_resample`, and fill/shift operations.
- Skipped columns and test-set refitting in `activeTrans`.

### Added

- Strict indexed `SData` and `SDataFrame` containers.
- Frozen TF-IDF vocabulary, fixed-width image summaries, stable statistical names.
- Approximate FPCA and optional tsfresh/B-spline/LOWESS feature transformers.
- Reproducible split utilities and a row-gap walk-forward cross-validator.
- Diagnostic feature-matrix quality reports and numerical validation.
- Editable installation via `pyproject.toml` and a Python-version test matrix.
- Expanded unit tests and runnable heterogeneous training example.
- Explicit compatibility guidance and evaluation leakage warnings.

### Deliberately out of scope

- Migration of the original Jupyter Notebook output/cells.
- Guaranteeing legacy `pyFDA` behavior on Python 3.13.
- Proving the mathematical/statistical equivalence of modernized transforms
  to experimental curves and coefficients in the 2017 thesis.
- Production-scale latency, memory or model-quality benchmarks.

Use [docs/compatibility.md](docs/compatibility.md) for migration behavior.
