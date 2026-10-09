# Machine Learning for Structured Data (MLSD)

A Python toolkit for mixing **ragged time series, unordered feature bags, images, text, and scalar columns** in a single scikit-learn workflow.

> **Project history:** MLSD began as the master's thesis work of Haoran Xue under the supervision of Dr. Franz Király at University College London (2017). The 0.2.x line modernizes the experimental research code, restores compatibility with current Python/scikit-learn and adds automated regression coverage. This is still a research-oriented library, not a verified production forecasting system.

![Original MLSD workflow](Workflow.png)

## Why it exists

A regular numerical DataFrame cannot naturally store an observation containing, for example, 12 hourly prices, 100 sensor readings, a product image and a text note as independently transformed columns. MLSD represents each *row* as a sample and each *column* as a typed `SData` container:

| Type | Intended content | Default transformer | Output |
|---|---|---|---|
| `Series` | Ordered sequences, potentially variable length | `BasicSeries` | 21 summary / derivative statistics |
| `Bag` | Unordered sets of numeric observations | `BasicBag` | 7 summary statistics |
| `Bag` + `mesh=True` | Mesh / ordered geometric samples | `BasicBag(mesh=True)` | 21 statistics including differences |
| `Text` | Raw text documents | `BasicText` | Learned sparse TF-IDF vocabulary |
| `Image` | 2-D / channel-last arrays | `BasicImage` | Dimensions and per-channel statistics |
| `NonS` | Ordinary scalar features | Identity / passthrough | Single column |

The derived features are merged by **exact row-index alignment**, with names such as `price__value_mean` and `news__bear`.

## Install

Python 3.10–3.13 is supported:

```bash
git clone https://github.com/xxxxxwater/Machine_Learning_For_Structured_Data.git
cd Machine_Learning_For_Structured_Data
python -m pip install -e .
python -m pip install -e '.[dev]'
python -m pytest -q
```

Optional backends:

```bash
python -m pip install -e '.[tsfresh,spline,lowess]'
```

The original `Dataset/growth.mat` and `Example.ipynb` remain for historical research reproducibility. They have **not** been fully ported or validated against today's dependencies.

## Quick start: fit training data, transform held-out data

```python
import numpy as np
from MLSD import SData, SDataFrame
from MLSD.model_selection import split_frame

frame = SDataFrame([
    SData([[1, 2, 3], [2, 3], [8, 9, 10], [9, 10]],
          column="price", dtype="Series"),
    SData(["down day", "down move", "up day", "up move"],
          column="headline", dtype="Text"),
])

X_train, y_train, X_test, y_test = split_frame(
    frame, np.array([0, 0, 1, 1]),
    test_size=0.25, shuffle=False,
)
features_train = X_train.fit_transform()
features_test = X_train.transform(X_test)

assert list(features_train.columns) == list(features_test.columns)
print(features_train.shape, features_test.shape)
```

For an end-to-end classifier demo with missing-value imputation and a temporal holdout, run:

```bash
python examples/train_mixed.py
```

**No leakage:** Always split *before* fitting the text vocabulary, dimensionality reducers, imputer or model. `frame.extracted_features` exists for convenience, but *fits on the entire frame* and should not be used on pre-split research/evaluation datasets.

## GitHub Packages: MLSD container

A ready-to-run **Linux AMD64** Python image is available through GitHub
Container Registry, built and smoke-tested from the immutable `v0.2.0`
release tag.

```bash
docker pull ghcr.io/xxxxxwater/mlsd-structured-data:v0.2.0
docker run --rm ghcr.io/xxxxxwater/mlsd-structured-data:v0.2.0
```

The image defaults to the synthetic mixed-data training demonstration.
See [GitHub Packages instructions](docs/packages.md) for using your own script,
authentication when a package is private, and the difference from the
wheel and source packages on the Release page.

## Time-series evaluation and gap

When labels look ahead across future bars, a naïve shuffled split—or even an ordinary holdout without a sufficient gap—may leak information:

```python
from MLSD.model_selection import PurgedWalkForwardSplit, take_rows

splitter = PurgedWalkForwardSplit(n_splits=3, gap=5, max_train_size=500)
for train_idx, val_idx in splitter.split(frame):
    training = take_rows(frame, train_idx)
    validation = take_rows(frame, val_idx)
    X_train = training.fit_transform()
    X_val = training.transform(validation)
    # Fit the predictive estimator on X_train and validate only on X_val.
```

Rows must already be ordered chronologically. `gap=5` excludes the five observations immediately preceding each validation block from the training window. **It does not** infer each target's true event end time; for irregular or overlapping label spans, you need label-aware purging beyond this row-gap splitter.

## Transformer API

`BasicSeries`, `BasicBag`, `BasicImage`, `BasicText`, `BsplineSeries`, `localRSeries`, `FPCA`, and `tsfreshSeries` implement `fit/transform`. Most expose `get_feature_names_out`. `activeTrans` can be used as a scikit-learn compatible wrapper around a single `SData` or a whole `SDataFrame`.

The optional `FPCA` name now represents an interpolated-grid PCA approximation. Its coefficients are **not directly comparable** to those from historical elastic functional-PCA implementations.

See [compatibility and migration notes](docs/compatibility.md) for behavior changes and known limitations.

## Quality and scope

- `tests/`: container validation, ragged summaries, all-column transformations, text vocabulary freezing, variable-sized images, temporal gaps, FPCA and optional backends.
- [GitHub Actions CI](.github/workflows/ci.yml): Python 3.10, 3.11, 3.12 and 3.13, pytest and wheel/sdist builds.
- This library provides **feature representations**, not calibrated probabilities or guaranteed financial predictions.
- Raw data, original workflow image and the historical Notebook remain intact.

## Licensing and attribution

MIT; see [LICENSE](LICENSE). Original thesis authorship is preserved in the repository's history and packaging metadata.
