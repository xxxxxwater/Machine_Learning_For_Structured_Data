# Cross-validation and leakage control

## Use a data-only view with scikit-learn

`SDataFrame.fit()` is a convenience feature-extraction API. scikit-learn's
`cross_val_score` distinguishes **data** from **estimators** and rejects
objects with a callable `fit` method as X. For direct cross-validation,
use the **non-estimator** `frame.as_sklearn()` view instead.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from MLSD import SData, SDataFrame, activeTrans

frame = SDataFrame([
    SData([[i % 2, i % 2 + 1, i % 2 + 2] for i in range(30)],
          column="signal")
])
y = np.arange(30) % 2
estimator = make_pipeline(activeTrans(), LogisticRegression(max_iter=200))
scores = cross_val_score(estimator, frame.as_sklearn(), y, cv=3)
print(scores)
```

`as_sklearn()` exposes an indexable `FrameSamples` view without `fit()`.
Row selection deep-copies observation data and transformers. Inside the
pipeline, `activeTrans` performs training-only fitting on the selected
fold and transforms held-out samples without refitting. This supports
`cross_val_score` and other tools that use compatible row indexing.

## Time-ordered labels: row gap or actual event end?

- `PurgedWalkForwardSplit(gap=k)` discards the final **k training rows**
  before each test fold. Use this when a fixed maximum lookahead (in bars)
  is known.
- `PurgedEventTimeSeriesSplit(label_end_times=...)` additionally discards
  every training sample whose *actual* label end timestamp overlaps the
  first evaluation timestamp. Use for irregular-duration labels.

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from MLSD.model_selection import PurgedEventTimeSeriesSplit

# The frame index must be chronological and unique, with UTC or other
# consistent datetime timezone; labels must use exactly the same index.
start = frame.index
ends = pd.Series(start + pd.Timedelta(hours=4), index=start)
cv = PurgedEventTimeSeriesSplit(ends, n_splits=3, gap=1)
scores = cross_val_score(estimator, frame.as_sklearn(), y, cv=cv)
```

For this example, make sure `frame.index` is a pandas `DatetimeIndex`;
the earlier toy frame's default `RangeIndex` is intentionally unsuitable.

### Precise overlap convention

An event interval includes its end time for purging purposes. A training
observation is retained only if:

```text
training_label_end < first_validation_observation_start
```

The end-time Series must have valid timestamps in matching timezone,
contain no `NaT`, and align exactly with X rows. Any event ending before
its own start is invalid. If purging empties a training fold, the
splitter raises `ValueError` instead of silently producing untrainable
folds. The model never trains on future rows after its validation fold.

## Important limitations

This prevents known **target-horizon overlap**, but not every source of
financial or time-series leakage. Cross-sectional preprocessing, universe
selection, normalization, corporate actions, feature timestamps, and
external as-of joins must themselves use information available at the
time of prediction. Group leakage (same entity appearing in both folds)
requires group-aware splitting that matches the research question.

`SDataFrame.transform()` rejects mismatched ordered column names or
per-column data types. A pandas `y` Series/DataFrame with the correct
length but misordered or unrelated index is rejected by `fit()` and
`split_frame()` instead of being paired positionally without warning.

The original `v0.2.0` release and published GHCR image remain unchanged;
this guide covers ongoing development on `master`.
