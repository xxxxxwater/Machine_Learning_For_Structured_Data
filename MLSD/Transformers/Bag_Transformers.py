"""Order-invariant bag statistics and optional order-sensitive mesh differences."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .Series_Transformers import STAT_NAMES, _as_rows, _stats


class BasicBag(BaseEstimator, TransformerMixin):
    """Seven summary statistics, or 21 including mesh finite differences.

    A bag is inherently unordered; mesh=True deliberately treats its input order
    as meaningful when deriving first and second differences.
    """

    def __init__(self, mesh=False, Dreduction=None):
        self.mesh = mesh
        self.Dreduction = Dreduction

    def fit(self, X, y=None):
        if self.Dreduction is not None:
            raise ValueError("BasicBag does not support Dreduction; use a pipeline")
        prefixes = ("value", "diff1", "diff2") if self.mesh else ("value",)
        self.feature_names_out_ = np.asarray(
            [f"{prefix}_{name}" for prefix in prefixes for name in STAT_NAMES],
            dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        rows, index = _as_rows(X)
        features = []
        for row in rows:
            a = np.asarray(row, dtype=float).ravel()
            result = _stats(a)
            if self.mesh:
                result += _stats(np.diff(a)) + _stats(np.diff(a, n=2))
            features.append(result)
        return pd.DataFrame(features, index=index, columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
