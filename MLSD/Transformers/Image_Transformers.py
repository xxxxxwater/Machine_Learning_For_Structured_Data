"""Compact image summaries, suitable for mixing image and tabular columns."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .Series_Transformers import _as_rows


class BasicImage(BaseEstimator, TransformerMixin):
    """Image dimensions and per-channel finite pixel statistics.

    Handles varying image sizes without expanding every pixel into a feature.
    Images must consistently be grayscale (2-D) or have the same channel count.
    """

    def __init__(self, Dreduction=None):
        self.Dreduction = Dreduction

    @staticmethod
    def _image(value):
        array = np.asarray(value, dtype=float)
        if array.ndim == 2:
            return array[:, :, None]
        if array.ndim != 3:
            raise ValueError("Expected a 2-D image or 3-D channel-last image")
        return array

    def fit(self, X, y=None):
        if self.Dreduction is not None:
            raise ValueError("BasicImage does not support Dreduction; use a pipeline")
        rows, _ = _as_rows(X)
        if not rows:
            raise ValueError("Cannot fit on an empty set of images")
        self.n_channels_ = self._image(rows[0]).shape[2]
        if self.n_channels_ < 1:
            raise ValueError("Images must have at least one channel")
        names = ["height", "width"]
        for channel in range(self.n_channels_):
            names.extend(f"c{channel}_{stat}" for stat in
                         ("min", "max", "mean", "std", "median"))
        self.feature_names_out_ = np.asarray(names, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        rows, index = _as_rows(X)
        features = []
        for value in rows:
            a = self._image(value)
            if a.shape[2] != self.n_channels_:
                raise ValueError("Inconsistent image channel count")
            row = [float(a.shape[0]), float(a.shape[1])]
            for channel in range(self.n_channels_):
                pixels = a[:, :, channel].ravel()
                pixels = pixels[np.isfinite(pixels)]
                if pixels.size:
                    row.extend([float(np.min(pixels)), float(np.max(pixels)),
                                float(np.mean(pixels)), float(np.std(pixels)),
                                float(np.median(pixels))])
                else:
                    row.extend([np.nan] * 5)
            features.append(row)
        return pd.DataFrame(features, index=index, columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
