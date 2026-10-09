"""Time-series feature transformers.

Only NumPy, pandas, SciPy and scikit-learn are required for the baseline
transformers. Optional backends are imported when they are used.
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


STAT_NAMES = ("min", "max", "mean", "std", "skew", "kurtosis", "median")


def _stats(values):
    """Seven scalar statistics; missing observations are ignored."""
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if not a.size:
        return [np.nan] * len(STAT_NAMES)
    spread = float(np.std(a))
    return [
        float(np.min(a)), float(np.max(a)), float(np.mean(a)), spread,
        float(skew(a)) if a.size >= 3 and spread > 0 else 0.0,
        float(kurtosis(a)) if a.size >= 4 and spread > 0 else 0.0,
        float(np.median(a)),
    ]


def _as_rows(X):
    if hasattr(X, "values") and hasattr(X, "index"):
        return list(X.values), pd.Index(X.index)
    rows = list(X)
    return rows, pd.RangeIndex(len(rows))


class BasicSeries(BaseEstimator, TransformerMixin):
    """21 stable statistics from each series and its first two differences."""

    def __init__(self, Dreduction=None):
        self.Dreduction = Dreduction

    def fit(self, X, y=None):
        if self.Dreduction is not None:
            raise ValueError("BasicSeries does not support Dreduction; use a pipeline")
        self.feature_names_out_ = np.asarray(
            [f"{prefix}_{name}" for prefix in ("value", "diff1", "diff2")
             for name in STAT_NAMES], dtype=object)
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        rows, index = _as_rows(X)
        features = []
        for row in rows:
            a = np.asarray(row, dtype=float).ravel()
            features.append(_stats(a) + _stats(np.diff(a)) + _stats(np.diff(a, n=2)))
        return pd.DataFrame(features, index=index, columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class tsfreshSeries(BaseEstimator, TransformerMixin):
    """Optional tsfresh backend; freeze selected features on training data."""

    def __init__(self, default_fc_parameters=None, n_jobs=0):
        self.default_fc_parameters = default_fc_parameters
        self.n_jobs = n_jobs

    @staticmethod
    def _long(X):
        rows, index = _as_rows(X)
        records = []
        for sample_id, row in enumerate(rows):
            series = pd.Series(row)
            for position, value in enumerate(series.to_numpy()):
                records.append((sample_id, position, value))
        if not records:
            raise ValueError("tsfresh requires at least one nonempty series")
        return pd.DataFrame(records, columns=["id", "time", "value"]), index

    def _extract(self, X):
        try:
            from tsfresh import extract_features
        except ImportError as exc:
            raise ImportError("Install MLSD[tsfresh] to use tsfreshSeries") from exc
        long, index = self._long(X)
        result = extract_features(
            long, column_id="id", column_sort="time",
            default_fc_parameters=self.default_fc_parameters,
            n_jobs=self.n_jobs, disable_progressbar=True)
        return result.reindex(range(len(index))).set_axis(index)

    def fit(self, X, y=None):
        features = self._extract(X).replace([np.inf, -np.inf], np.nan)
        if y is not None:
            try:
                from tsfresh import select_features
            except ImportError as exc:
                raise ImportError("Install MLSD[tsfresh] to select features") from exc
            target = pd.Series(np.asarray(y), index=features.index)
            features = select_features(features.fillna(0), target)
        self.feature_names_out_ = np.asarray(features.columns, dtype=object)
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        return self._extract(X).reindex(columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class BsplineSeries(BaseEstimator, TransformerMixin):
    """Per-series B-spline coefficients using a fixed basis size."""

    def __init__(self, degrees=3, knots=6):
        self.degrees = degrees
        self.knots = knots

    def fit(self, X, y=None):
        if self.degrees < 0 or self.knots < self.degrees + 1:
            raise ValueError("knots must be at least degrees + 1")
        self.feature_names_out_ = np.array(
            [f"bspline_{i}" for i in range(self.knots)], dtype=object)
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        try:
            from patsy import dmatrix
        except ImportError as exc:
            raise ImportError("Install MLSD[spline] to use BsplineSeries") from exc
        rows, index = _as_rows(X)
        result = []
        for row in rows:
            a = np.asarray(row, dtype=float).ravel()
            if a.size < self.knots or not np.isfinite(a).all():
                raise ValueError("B-spline series must be finite and at least knots long")
            t = np.linspace(0, 1, a.size)
            basis = np.asarray(dmatrix(
                "bs(t, df=df, degree=degree, include_intercept=False) - 1",
                {"t": t, "df": self.knots, "degree": self.degrees}))
            result.append(np.linalg.lstsq(basis, a, rcond=None)[0])
        return pd.DataFrame(result, index=index, columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class localRSeries(BaseEstimator, TransformerMixin):
    """LOWESS summaries (requires the optional statsmodels dependency)."""

    def __init__(self, fraction=0.3):
        self.fraction = fraction

    def fit(self, X, y=None):
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must lie in (0, 1]")
        self.feature_names_out_ = np.asarray(
            ["smooth_" + name for name in STAT_NAMES], dtype=object)
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError as exc:
            raise ImportError("Install MLSD[lowess] to use localRSeries") from exc
        rows, index = _as_rows(X)
        result = []
        for row in rows:
            a = np.asarray(row, dtype=float).ravel()
            valid = np.isfinite(a)
            if valid.sum() < 2:
                result.append([np.nan] * len(STAT_NAMES))
            else:
                curve = lowess(a[valid], np.flatnonzero(valid), frac=self.fraction,
                               return_sorted=False)
                result.append(_stats(curve))
        return pd.DataFrame(result, index=index, columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class FPCA(BaseEstimator, TransformerMixin):
    """Approximate functional PCA by interpolation onto a common time grid.

    Fit only on training samples to avoid test-set information leakage.
    """

    def __init__(self, n_components=2, n_points=64):
        self.n_components = n_components
        self.n_points = n_points

    def _matrix(self, X):
        rows, index = _as_rows(X)
        matrix = []
        for row in rows:
            a = np.asarray(row, dtype=float).ravel()
            valid = np.isfinite(a)
            if valid.sum() < 2:
                raise ValueError("FPCA requires at least two finite samples per series")
            xp = np.linspace(0.0, 1.0, len(a))[valid]
            matrix.append(np.interp(np.linspace(0, 1, self.n_points), xp, a[valid]))
        return np.asarray(matrix), index

    def fit(self, X, y=None):
        if self.n_points < 2 or self.n_components < 1:
            raise ValueError("n_points must be >= 2 and n_components >= 1")
        matrix, _ = self._matrix(X)
        self.model_ = PCA(n_components=self.n_components).fit(matrix)
        self.feature_names_out_ = np.asarray(
            [f"fpca_{i}" for i in range(self.n_components)], dtype=object)
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        matrix, index = self._matrix(X)
        return pd.DataFrame(self.model_.transform(matrix), index=index,
                            columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
