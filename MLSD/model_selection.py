"""Reproducible train/test selection for heterogeneous structured observations."""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from .StructureData import SData
from .StructureDataFrame import SDataFrame


def take_rows(frame, positions):
    """Select rows by position while preserving row labels and feature schemas."""
    if not isinstance(frame, SDataFrame):
        raise TypeError("frame must be an SDataFrame")
    ids = np.arange(len(frame))[positions]
    ids = np.atleast_1d(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate row positions would create ambiguous indexes")
    data = [
        SData(item.values[ids], index=item.index.take(ids),
              dtype=item.dtype, column=item.column,
              transformer=deepcopy(item.transformer))
        for item in frame.data
    ]
    return SDataFrame(data, columns=frame.columns, transformers=deepcopy(frame.transformers))


def split_frame(frame, y, test_size=0.25, shuffle=True, random_state=42):
    """Return X_train, y_train, X_test, y_test without fitting on held-out data.

    Set shuffle=False for ordered observations. For overlapping time-series labels,
    use PurgedWalkForwardSplit with a gap corresponding to the label horizon.
    """
    if not isinstance(frame, SDataFrame):
        raise TypeError("frame must be an SDataFrame")
    if isinstance(y, pd.Series) and not y.index.equals(frame.index):
        raise ValueError("y index must match the frame index in the same order")
    labels = np.asarray(y)
    count = len(frame)
    if labels.ndim == 0 or len(labels) != count:
        raise ValueError("y must have one label per observation")
    if isinstance(test_size, (float, np.floating)):
        if not 0 < test_size < 1:
            raise ValueError("test_size fraction must be between zero and one")
        n_test = int(np.ceil(test_size * count))
    elif isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
    else:
        raise TypeError("test_size must be an integer or a fraction")
    if not 0 < n_test < count:
        raise ValueError("test_size must leave both training and testing rows")
    ids = np.arange(count)
    if shuffle:
        ids = np.random.default_rng(random_state).permutation(ids)
    train_ids, test_ids = ids[:-n_test], ids[-n_test:]
    return (
        take_rows(frame, train_ids), labels[train_ids],
        take_rows(frame, test_ids), labels[test_ids],
    )


class PurgedWalkForwardSplit(BaseCrossValidator):
    """Expanding/rolling train windows strictly before validation windows.

    gap removes the last gap training rows before each test fold. Choose gap >=
    the maximum known forward-label horizon (in rows). Inputs must be sorted
    chronologically. This does not solve arbitrary overlapping event labels.
    """

    def __init__(self, n_splits=3, gap=0, max_train_size=None):
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if not isinstance(self.n_splits, int) or self.n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")
        if not isinstance(self.gap, int) or self.gap < 0:
            raise ValueError("gap must be a nonnegative integer")
        if (self.max_train_size is not None and
                (not isinstance(self.max_train_size, int) or self.max_train_size < 1)):
            raise ValueError("max_train_size must be a positive integer")
        count = len(X)
        if count < self.n_splits + 1:
            raise ValueError("Not enough samples for the requested n_splits")
        if hasattr(X, "index") and isinstance(X.index, pd.DatetimeIndex):
            if not X.index.is_monotonic_increasing:
                raise ValueError("Time series rows must be ordered by time")
        test_count = count // (self.n_splits + 1)
        for i in range(self.n_splits):
            test_start = count - (self.n_splits - i) * test_count
            test_stop = test_start + test_count
            train_stop = test_start - self.gap
            if train_stop <= 0:
                raise ValueError("gap leaves no training rows for the first fold")
            train_start = (max(0, train_stop - self.max_train_size)
                           if self.max_train_size is not None else 0)
            yield np.arange(train_start, train_stop), np.arange(test_start, test_stop)
