"""Reproducible train/test selection for heterogeneous structured observations."""
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from .StructureDataFrame import SDataFrame


def take_rows(frame, positions):
    """Select rows by position while preserving row labels and feature schemas."""
    if not isinstance(frame, SDataFrame):
        raise TypeError("frame must be an SDataFrame")
    return frame.take(positions, axis=0)


def split_frame(frame, y, test_size=0.25, shuffle=True, random_state=42):
    """Return X_train, y_train, X_test, y_test without fitting on held-out data.

    Set shuffle=False for ordered observations. For overlapping time-series labels,
    use PurgedWalkForwardSplit with a gap corresponding to the label horizon.
    """
    if not isinstance(frame, SDataFrame):
        raise TypeError("frame must be an SDataFrame")
    if isinstance(y, (pd.Series, pd.DataFrame)) and not y.index.equals(frame.index):
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


class PurgedEventTimeSeriesSplit(PurgedWalkForwardSplit):
    """Walk-forward validation that purges *actual* overlapping label horizons.

    Unlike a fixed row gap, the `label_end_times` Series provides a timestamp
    for when every row's forward-looking target finishes. Training rows whose
    target remains unresolved at the first validation sample are discarded.

    This is an expanding/rolling **past-only** CV: it never trains on
    observations after the validation start. An embargo on future training
    observations is therefore unnecessary. Users must still ensure that
    feature engineering itself does not look into the future.
    """

    def __init__(self, label_end_times, n_splits=3, gap=0, max_train_size=None):
        super().__init__(
            n_splits=n_splits, gap=gap, max_train_size=max_train_size)
        self.label_end_times = label_end_times

    def split(self, X, y=None, groups=None):
        if not hasattr(X, "index") or not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("X must have a DatetimeIndex for event-aware purging")
        starts = X.index
        if starts.hasnans or not starts.is_unique or not starts.is_monotonic_increasing:
            raise ValueError("X index must contain unique chronological increasing timestamps")
        ends = self.label_end_times
        if not isinstance(ends, pd.Series):
            raise TypeError("label_end_times must be a pandas Series indexed by X.index")
        if not ends.index.equals(starts):
            raise ValueError("label_end_times index must match X index and row order")
        if not pd.api.types.is_datetime64_any_dtype(ends.dtype):
            raise TypeError("label_end_times values must be datetime timestamps")
        if ends.isna().any():
            raise ValueError("label_end_times contains missing (NaT) end timestamps")
        if ends.dt.tz != starts.tz:
            raise ValueError("label_end_times timezone must match X.index timezone")
        if bool((ends.array < starts.array).any()):
            raise ValueError("A label end cannot be earlier than its observation start")

        for train_idx, test_idx in super().split(X, y, groups):
            first_test_time = starts[test_idx[0]]
            eligible = (ends.iloc[train_idx] < first_test_time).to_numpy(dtype=bool)
            purged_train = train_idx[eligible]
            if not len(purged_train):
                raise ValueError("Event-time purge left no training samples for a fold")
            yield purged_train, test_idx
