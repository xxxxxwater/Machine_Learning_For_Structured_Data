import numpy as np
import pandas as pd
import pytest

from MLSD import SData, SDataFrame
from MLSD.model_selection import PurgedWalkForwardSplit, split_frame, take_rows


def make_frame(n=18):
    index = pd.date_range("2025-01-01", periods=n, freq="h")
    column = SData([[i, i + 1] for i in range(n)],
                   index=index, column="price", dtype="Series")
    return SDataFrame([column])


def test_take_rows_preserves_dtype_order_and_index():
    frame = make_frame()
    sampled = take_rows(frame, [5, 1, 2])
    assert sampled.index.tolist() == frame.index[[5, 1, 2]].tolist()
    assert sampled.iloc(0, 0).tolist() == [5, 6]
    assert sampled.columns == frame.columns
    assert sampled.dtype.tolist() == ["Series"]
    with pytest.raises(ValueError, match="Duplicate"):
        take_rows(frame, [1, 1])


def test_holdout_respects_chronological_order_and_corresponding_labels():
    frame = make_frame()
    labels = pd.Series(np.arange(18), index=frame.index)
    train, y_train, test, y_test = split_frame(
        frame, labels, test_size=4, shuffle=False)
    assert len(train) == 14
    assert len(test) == 4
    assert list(y_train) == list(range(14))
    assert list(y_test) == list(range(14, 18))
    assert train.index[-1] < test.index[0]
    assert train.fit_transform().shape == (14, 21)
    assert train.transform(test).shape == (4, 21)


def test_split_rejects_misaligned_targets_and_bad_sizes():
    frame = make_frame()
    with pytest.raises(ValueError, match="y index"):
        split_frame(frame, pd.Series(np.arange(18)))
    with pytest.raises(ValueError, match="test_size"):
        split_frame(frame, np.arange(18), test_size=18)


def test_purged_folds_have_no_overlap_or_forward_leakage():
    frame = make_frame(30)
    splitter = PurgedWalkForwardSplit(n_splits=3, gap=2)
    folds = list(splitter.split(frame))
    assert len(folds) == 3
    for train, test in folds:
        assert len(set(train) & set(test)) == 0
        assert train[-1] + 2 < test[0]
        assert np.all(np.diff(train) == 1)
        assert np.all(np.diff(test) == 1)
    assert folds[0][0][0] == 0
    assert folds[1][0].size > folds[0][0].size


def test_rolling_window_bound_and_errors():
    frame = make_frame(30)
    folds = list(PurgedWalkForwardSplit(n_splits=2, gap=1,
                                        max_train_size=5).split(frame))
    assert all(len(train) <= 5 for train, _ in folds)
    with pytest.raises(ValueError, match="no training rows"):
        list(PurgedWalkForwardSplit(n_splits=3, gap=100).split(frame))
    reverse = make_frame()
    reverse.index = reverse.index[::-1]
    with pytest.raises(ValueError, match="ordered by time"):
        list(PurgedWalkForwardSplit().split(reverse))
