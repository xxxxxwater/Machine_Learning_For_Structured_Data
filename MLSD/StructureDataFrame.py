"""Dataframe-like container for heterogeneous structured feature columns."""
from copy import deepcopy
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from .StructureData import SData


class _FrameILoc:
    """Position-based row/column selection; legacy iloc(row, col) is retained."""

    def __init__(self, frame):
        self.frame = frame

    def __call__(self, row, col):
        if not isinstance(row, Integral) or not isinstance(col, Integral):
            raise TypeError("iloc(row, col) expects integer positions")
        return self.frame.data[col][row]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("iloc expects at most two dimensions")
            row, col = key
            if isinstance(row, Integral) and isinstance(col, Integral):
                return self(row, col)
            return self.frame.take(col, axis=1).take(row, axis=0)
        if isinstance(key, Integral):
            return [item[key] for item in self.frame.data]
        return self.frame.take(key, axis=0)


class _SamplesILoc:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, positions):
        selected = self.samples.frame.iloc[positions]
        if isinstance(selected, SDataFrame):
            return FrameSamples(selected)
        return selected


class FrameSamples:
    """Indexable data-only view compatible with sklearn cross-validation.

    SDataFrame itself implements fit(), so sklearn correctly treats it as an
    estimator rather than as indexable X. This wrapper deliberately does
    *not* expose fit() and preserves independent copies of selected rows.
    """

    def __init__(self, frame):
        if not isinstance(frame, SDataFrame):
            raise TypeError("FrameSamples requires an SDataFrame")
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    @property
    def shape(self):
        return self.frame.shape

    @property
    def index(self):
        return self.frame.index

    @property
    def iloc(self):
        return _SamplesILoc(self)

    def take(self, indices, axis=0):
        return FrameSamples(self.frame.take(indices, axis=axis))

    def __repr__(self):
        return f"FrameSamples(shape={self.shape!r})"


class SDataFrame:
    """Combine typed SData columns into an aligned scikit-learn feature matrix."""

    def __init__(self, data, columns=None, index=None, transformers=None):
        self.data = list(data)
        if not self.data:
            raise ValueError("SDataFrame requires at least one SData column")
        if not all(isinstance(item, SData) for item in self.data):
            raise TypeError("Every column must be an SData instance")
        self.index = pd.Index(self.data[0].index if index is None else index)
        if not self.index.is_unique:
            raise ValueError("index labels must be unique")
        for item in self.data:
            if len(item) != len(self.index) or not item.index.equals(self.index):
                raise ValueError("All SData columns must have identical ordered indexes")
        names = ([item.column if item.column is not None else f"feature_{i}"
                  for i, item in enumerate(self.data)] if columns is None else columns)
        if len(names) != len(self.data):
            raise ValueError("columns length must match data length")
        self.columns = list(names)
        if len(set(map(str, self.columns))) != len(self.columns):
            raise ValueError("Column names must be unique")
        if transformers is not None and len(transformers) != len(self.data):
            raise ValueError("transformers length must match data length")
        self.transformers = list(transformers) if transformers is not None else [
            item.transformer for item in self.data]
        self.len = len(self.index)
        self.dtype = np.asarray([item.dtype for item in self.data])
        self.size = np.asarray([item.size for item in self.data], dtype=object)

    def __len__(self):
        return len(self.index)

    @property
    def shape(self):
        return (len(self.index), len(self.data))

    @property
    def iloc(self):
        return _FrameILoc(self)

    def as_sklearn(self):
        """Provide indexable samples for sklearn cross_val_score/GridSearchCV."""
        return FrameSamples(self)

    def take(self, indices, axis=0):
        """Return independent selected rows (axis=0) or columns (axis=1).

        Compatible with scikit-learn's private row selection used in CV
        while retaining nested structured observations and their dtypes.
        """
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        positions = np.arange(self.shape[axis])[indices]
        positions = np.atleast_1d(positions)
        if positions.dtype.kind not in ("i", "u"):
            raise TypeError("take requires integer positions or a boolean mask")
        if len(np.unique(positions)) != len(positions):
            raise ValueError("Duplicate positions would create ambiguous labels")
        if axis == 0:
            new_index = self.index.take(positions)
            new_data = []
            for item in self.data:
                # Rebuild typed objects to avoid train/test aliasing.
                new_data.append(SData(
                    deepcopy(item.values[positions]), index=new_index,
                    column=item.column, dtype=item.dtype,
                    transformer=deepcopy(item.transformer)))
            return SDataFrame(new_data, columns=self.columns,
                              index=new_index, transformers=deepcopy(self.transformers))
        new_data = [deepcopy(self.data[i]) for i in positions]
        return SDataFrame(
            new_data, columns=[self.columns[i] for i in positions],
            index=self.index.copy(),
            transformers=[deepcopy(self.transformers[i]) for i in positions])

    def __str__(self):
        return f"SDataFrame(n_samples={len(self)}, columns={self.columns!r})"

    __repr__ = __str__

    def __getitem__(self, label):
        """Return column values for a row identified by its index label."""
        position = self.index.get_loc(label)
        return [item[position] for item in self.data]

    def __getattr__(self, name):
        if "columns" in self.__dict__ and name in self.columns:
            return self.data[self.columns.index(name)]
        raise AttributeError(name)

    def _stat(self, name):
        if any(item.dtype not in ("Series", "Bag") for item in self.data):
            raise TypeError("Aggregate statistics require only Series/Bag columns")
        return pd.DataFrame({str(col): getattr(item, name)()
                             for col, item in zip(self.columns, self.data)},
                            index=self.index)

    def min(self):
        return self._stat("min")

    def max(self):
        return self._stat("max")

    def mean(self):
        return self._stat("mean")

    def std(self):
        return self._stat("std")

    def shift(self, n=1):
        for item in self.data:
            item.shift(n)
        self.index = self.data[0].index
        self.len = len(self.index)
        self.size = np.asarray([item.size for item in self.data], dtype=object)
        return self

    def inner_shift(self, n=1):
        for item in self.data:
            if item.dtype == "Series":
                item.values = item.inner_shift(n)
        return self

    def resample(self, freq, func="mean"):
        for item in self.data:
            if item.dtype == "Series":
                item.resample(freq, func)
        return self

    def C_resample(self, freq, func="mean"):
        result = deepcopy(self)
        return result.resample(freq, func)

    def fillna(self, value):
        for item in self.data:
            if item.dtype in ("Series", "Bag"):
                item.fillna(value)
        return self

    def ffill(self, limit=None):
        for item in self.data:
            if item.dtype in ("Series", "Bag"):
                item.ffill(limit)
        return self

    def bfill(self, limit=None):
        for item in self.data:
            if item.dtype in ("Series", "Bag"):
                item.bfill(limit)
        return self

    def join(self, new_sdata, column=None):
        """Return a new frame with a column aligned by its row labels."""
        if not isinstance(new_sdata, SData):
            raise TypeError("join expects SData")
        if not new_sdata.index.equals(self.index):
            raise ValueError("Joined column must have identical ordered indexes")
        name = column if column is not None else (
            new_sdata.column if new_sdata.column is not None
            else f"feature_{len(self.data)}")
        return SDataFrame(self.data + [new_sdata], columns=self.columns + [name],
                          index=self.index, transformers=self.transformers +
                          [new_sdata.transformer])

    def fit(self, X=None, y=None):
        """Fit independent cloned transformers on this training frame only."""
        if X is not None and X is not self:
            raise ValueError("Call frame.fit(y=...) on the frame being fitted")
        if isinstance(y, (pd.Series, pd.DataFrame)) and not y.index.equals(self.index):
            raise ValueError("y index must align with the frame index in the same order")
        if y is not None and len(y) != len(self):
            raise ValueError("y length must equal the number of samples")
        fitted = []
        for item, transformer in zip(self.data, self.transformers):
            if transformer is None:
                fitted.append(None)
                continue
            try:
                estimator = clone(transformer)
            except TypeError:
                estimator = deepcopy(transformer)
            estimator.fit(item, y)
            fitted.append(estimator)
        self.fitted_transformers_ = fitted
        self.fitted_dtypes_ = tuple(item.dtype for item in self.data)
        return self

    def transform(self, X=None):
        """Transform a frame using training-only fitted estimators."""
        if not hasattr(self, "fitted_transformers_"):
            raise NotFittedError("Call fit() before transform()")
        frame = self if X is None else X
        if not isinstance(frame, SDataFrame):
            raise TypeError("transform expects an SDataFrame")
        if frame.columns != self.columns:
            raise ValueError("Feature columns differ from those seen during fit")
        if len(frame.data) != len(self.fitted_transformers_):
            raise ValueError("Transformer count and column count differ")
        if tuple(item.dtype for item in frame.data) != self.fitted_dtypes_:
            raise ValueError("Column dtype schema differs from training data")
        parts = []
        for name, item, transformer in zip(
                frame.columns, frame.data, self.fitted_transformers_):
            if transformer is None:
                part = pd.DataFrame({"value": item.values}, index=frame.index)
            else:
                part = transformer.transform(item)
                if not isinstance(part, pd.DataFrame):
                    part = pd.DataFrame(part, index=frame.index)
                if not part.index.equals(frame.index):
                    raise ValueError(f"Transformer {name!r} changed the row index")
            part = part.copy()
            part.columns = [f"{name}__{column}" for column in part.columns]
            parts.append(part)
        return pd.concat(parts, axis=1).reindex(frame.index)

    def get_feature_names_out(self, input_features=None):
        """Return trained, column-prefixed feature names without refitting."""
        if not hasattr(self, "fitted_transformers_"):
            raise NotFittedError("Call fit() before get_feature_names_out()")
        names = []
        for column, transformer in zip(self.columns, self.fitted_transformers_):
            if transformer is None:
                features = ["value"]
            elif hasattr(transformer, "get_feature_names_out"):
                features = transformer.get_feature_names_out()
            else:
                raise AttributeError(
                    f"Transformer for {column!r} has no get_feature_names_out()")
            names.extend(f"{column}__{name}" for name in features)
        return np.asarray(names, dtype=object)

    def fit_transform(self, X=None, y=None):
        return self.fit(X=X, y=y).transform()

    @property
    def extracted_features(self):
        """Fit on all observations; use explicit fit/transform for evaluation."""
        return self.fit_transform()

    def apply(self, func):
        return [item.apply(func) for item in self.data]
