"""Containers for individual columns of structured observations."""
from copy import deepcopy

import numpy as np
import pandas as pd

from .Transformers import BasicBag, BasicImage, BasicSeries, BasicText


class SData:
    """A column of ragged Series, Bag, Image, Text, or scalar (NonS) values.

    Each observation occupies one slot, even when samples have different
    lengths. The index must be unique so feature matrices can align safely.
    """

    _TYPES = ("Series", "Bag", "Image", "Text", "NonS")

    def __init__(self, x, index=None, column=None, dtype="Series", transformer=None):
        if dtype not in self._TYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; expected {self._TYPES}")
        if not isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            raise TypeError("x must be a sequence of observations")
        observations = list(x)
        self.values = np.empty(len(observations), dtype=object)
        for j, item in enumerate(observations):
            if dtype in ("Series", "Bag"):
                self.values[j] = item.copy() if isinstance(item, pd.Series) else pd.Series(item)
            elif dtype == "Image":
                self.values[j] = np.asarray(item)
            else:
                self.values[j] = item
        self.index = pd.Index(range(len(self.values)) if index is None else index)
        if len(self.index) != len(self.values):
            raise ValueError("index length must match the number of observations")
        if not self.index.is_unique:
            raise ValueError("index labels must be unique")
        self.column = column
        self.dtype = dtype
        defaults = {"Series": BasicSeries, "Bag": BasicBag,
                    "Image": BasicImage, "Text": BasicText}
        self.transformer = transformer if transformer is not None else (
            defaults[dtype]() if dtype in defaults else None)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def __str__(self):
        return "SData"

    def __repr__(self):
        return f"SData(dtype={self.dtype!r}, n_samples={len(self)}, column={self.column!r})"

    @property
    def p_dtype(self):
        return self.values.dtype

    @property
    def size(self):
        return [len(value) if hasattr(value, "__len__") else 1 for value in self.values]

    def _require_numeric(self):
        if self.dtype not in ("Series", "Bag"):
            raise TypeError(f"{self.dtype} does not support numeric series operations")

    def _aggregate(self, func):
        self._require_numeric()
        return np.asarray([func(np.asarray(i, dtype=float)) for i in self.values])

    def min(self):
        return self._aggregate(np.min)

    def max(self):
        return self._aggregate(np.max)

    def std(self):
        return self._aggregate(np.std)

    def mean(self):
        return self._aggregate(np.mean)

    def append(self, new_row, index=None):
        """Append one observation; optionally supply its unique row label."""
        label = len(self.values) if index is None else index
        if label in self.index:
            raise ValueError(f"Duplicate index label: {label!r}")
        one = SData([new_row], index=[label], column=self.column,
                    dtype=self.dtype, transformer=self.transformer)
        self.values = np.concatenate([self.values, one.values])
        self.index = self.index.append(one.index)
        return self

    def reindex(self):
        self.index = pd.RangeIndex(len(self.values))
        return self

    def shift(self, n=1):
        """Lag observations n rows, discarding rows without aligned targets."""
        if not isinstance(n, (int, np.integer)):
            raise TypeError("n must be an integer")
        if n > 0:
            self.values, self.index = self.values[:-n], self.index[n:]
        elif n < 0:
            self.values, self.index = self.values[-n:], self.index[:n]
        return self

    def inner_shift(self, n=1):
        if self.dtype != "Series":
            raise TypeError("inner_shift only supports Series")
        shifted = np.empty(len(self), dtype=object)
        for j, series in enumerate(self.values):
            shifted[j] = series.shift(n)
        return shifted

    def fillna(self, value):
        self._require_numeric()
        for j, series in enumerate(self.values):
            self.values[j] = series.fillna(value)
        return self

    def ffill(self, x=None):
        self._require_numeric()
        for j, series in enumerate(self.values):
            self.values[j] = series.ffill(limit=x)
        return self

    def bfill(self, x=None):
        self._require_numeric()
        for j, series in enumerate(self.values):
            self.values[j] = series.bfill(limit=x)
        return self

    def split_train_test(self, y, test_ratio=0.3, random_state=42, shuffle=True):
        """Split without fitting a transformer on any held-out data."""
        if isinstance(y, (pd.Series, pd.DataFrame)) and not y.index.equals(self.index):
            raise ValueError("y index must match the SData index in the same order")
        labels = np.asarray(y)
        if labels.ndim == 0 or len(labels) != len(self):
            raise ValueError("y must have one label per observation")
        if not 0 < test_ratio < 1 or len(self) < 2:
            raise ValueError("test_ratio must be in (0, 1) and need >= 2 samples")
        order = np.arange(len(self))
        if shuffle:
            order = np.random.default_rng(random_state).permutation(order)
        test_count = max(1, min(len(self) - 1, int(np.ceil(test_ratio * len(self)))))
        train_ids, test_ids = order[:-test_count], order[-test_count:]
        def subset(ids):
            return SData(self.values[ids], index=self.index.take(ids),
                         column=self.column, dtype=self.dtype,
                         transformer=deepcopy(self.transformer))
        return subset(train_ids), labels[train_ids], subset(test_ids), labels[test_ids]

    @property
    def extracted_features(self):
        """Convenience fitting on all rows; use fit/transform for held-out tests."""
        if self.dtype == "NonS":
            return pd.DataFrame({"value": self.values}, index=self.index)
        return self.transformer.fit_transform(self)

    def resample(self, freq, func="mean"):
        if self.dtype != "Series":
            raise TypeError("Only Series data can be resampled")
        result = []
        for series in self.values:
            if not isinstance(series.index, pd.DatetimeIndex):
                raise TypeError("resample requires a DatetimeIndex for each series")
            result.append(series.resample(freq).agg(func))
        # Resampling is atomic: one invalid observation must not leave half
        # of the container transformed.
        for j, new_series in enumerate(result):
            self.values[j] = new_series
        return self

    def C_resample(self, freq, func="mean"):
        clone = SData(self.values, index=self.index.copy(), column=self.column,
                      dtype=self.dtype, transformer=deepcopy(self.transformer))
        return clone.resample(freq, func)

    def apply(self, func):
        return np.asarray([func(value) for value in self.values])

    def plot(self):
        import matplotlib.pyplot as plt
        self._require_numeric()
        for series in self.values:
            plt.plot(series)
        plt.show()
