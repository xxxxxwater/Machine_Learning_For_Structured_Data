# Changelog

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
