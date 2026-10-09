"""Lightweight feature-matrix quality diagnostics for structured ML."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureAudit:
    n_rows: int
    n_columns: int
    missing: dict
    infinite: dict
    constant_columns: tuple
    non_numeric_columns: tuple

    @property
    def is_finite_numeric(self):
        return (not self.missing and not self.infinite
                and not self.non_numeric_columns)


def audit_features(features):
    """Inspect feature columns without silently dropping bad or sparse values.

    Empty inputs, duplicate row labels and duplicate feature names are rejected.
    Missing and infinite cells are reported separately; legitimate constant
    features are identified rather than silently removed.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if features.empty:
        raise ValueError("Cannot audit an empty feature matrix")
    if not features.columns.is_unique or not features.index.is_unique:
        raise ValueError("Feature columns and row indexes must be unique")
    missing, infinite, constant, non_numeric = {}, {}, [], []
    for name in features.columns:
        series = features[name]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            non_numeric.append(str(name))
            continue
        values = series.to_numpy(dtype=float, na_value=np.nan)
        n_missing = int(np.isnan(values).sum())
        n_infinite = int(np.isinf(values).sum())
        if n_missing:
            missing[str(name)] = n_missing
        if n_infinite:
            infinite[str(name)] = n_infinite
        valid = values[np.isfinite(values)]
        if valid.size and np.all(valid == valid[0]):
            constant.append(str(name))
    return FeatureAudit(
        n_rows=len(features), n_columns=len(features.columns),
        missing=missing, infinite=infinite,
        constant_columns=tuple(constant),
        non_numeric_columns=tuple(non_numeric),
    )


def require_numeric_features(features, allow_missing=True):
    """Fail clearly on nonnumeric/infinite features before estimator training."""
    report = audit_features(features)
    if report.non_numeric_columns:
        raise ValueError(f"Nonnumeric features: {report.non_numeric_columns}")
    if report.infinite:
        raise ValueError(f"Infinite feature values: {report.infinite}")
    if not allow_missing and report.missing:
        raise ValueError(f"Missing feature values: {report.missing}")
    return report
