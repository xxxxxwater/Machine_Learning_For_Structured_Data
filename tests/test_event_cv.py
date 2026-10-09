"""Regression coverage for event-label leakage in chronological CV."""
import numpy as np
import pandas as pd
import pytest

from MLSD import SData, SDataFrame
from MLSD.model_selection import PurgedEventTimeSeriesSplit


def _inputs(n=32):
    starts = pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC")
    frame = SDataFrame([SData(
        [[float(i), float(i + 1), float(i + 2)] for i in range(n)],
        index=starts, column="price")])
    return frame, starts


def test_purge_uses_actual_label_end_times_not_just_row_gap():
    frame, starts = _inputs()
    durations = np.ones(len(frame), dtype=int)
    durations[7:13] = 15
    ends = pd.Series(starts + pd.to_timedelta(durations, unit="h"), index=starts)
    cv = PurgedEventTimeSeriesSplit(ends, n_splits=3, gap=0)
    folds = list(cv.split(frame))
    assert len(folds) == 3
    for train_ids, test_ids in folds:
        assert len(train_ids) >= 1
        assert np.all(train_ids < test_ids[0])
        assert all(ends.iloc[i] < starts[test_ids[0]] for i in train_ids)
    first_train, first_test = folds[0]
    assert first_test[0] == 8
    assert not any(i in first_train for i in range(7, 8))


def test_purge_combines_explicit_row_gap_with_real_event_end():
    frame, starts = _inputs()
    ends = pd.Series(starts + pd.Timedelta(hours=1), index=starts)
    cv = PurgedEventTimeSeriesSplit(ends, n_splits=3, gap=2)
    for train_ids, test_ids in cv.split(frame):
        assert train_ids[-1] + 2 < test_ids[0]


def test_rejects_misordered_and_invalid_event_end_labels():
    frame, starts = _inputs()
    valid = pd.Series(starts + pd.Timedelta(hours=2), index=starts)
    bad_index = valid.iloc[::-1]
    with pytest.raises(ValueError, match="index"):
        list(PurgedEventTimeSeriesSplit(bad_index).split(frame))
    bad_missing = valid.copy()
    bad_missing.iloc[5] = pd.NaT
    with pytest.raises(ValueError, match="missing|NaT"):
        list(PurgedEventTimeSeriesSplit(bad_missing).split(frame))
    bad_early = valid.copy()
    bad_early.iloc[5] = starts[4]
    with pytest.raises(ValueError, match="earlier|end"):
        list(PurgedEventTimeSeriesSplit(bad_early).split(frame))


def test_requires_chronological_datetime_index_and_timestamp_end_dates():
    frame, starts = _inputs()
    ends = pd.Series(starts + pd.Timedelta(hours=1), index=starts)
    out_of_order = frame.take([2, 0, 1] + list(range(3, len(frame))))
    with pytest.raises(ValueError, match="chronological|increasing"):
        list(PurgedEventTimeSeriesSplit(ends).split(out_of_order))
    not_datetime = SDataFrame([SData([[1, 2, 3]] * 32)])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        list(PurgedEventTimeSeriesSplit(ends).split(not_datetime))
    with pytest.raises(TypeError, match="Series"):
        list(PurgedEventTimeSeriesSplit(ends.to_numpy()).split(frame))


def test_refuses_fold_with_zero_nonoverlapping_training_labels():
    frame, starts = _inputs()
    # All training events finish after the first validation window begins.
    ends = pd.Series(starts[-1] + pd.Timedelta(hours=5), index=starts)
    with pytest.raises(ValueError, match="no training|purge"):
        list(PurgedEventTimeSeriesSplit(ends, n_splits=3).split(frame))
