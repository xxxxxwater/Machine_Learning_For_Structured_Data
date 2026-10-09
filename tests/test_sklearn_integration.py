"""Integration regression tests for scikit-learn and schema boundaries."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from MLSD import SData, SDataFrame, activeTrans
from MLSD.model_selection import PurgedWalkForwardSplit


def _frame(n=30):
    """Each row contains a ragged but class-informative sequence."""
    index = pd.date_range("2026-01-01", periods=n, freq="h")
    prices = SData(
        [np.array([float(i % 2), float(i % 2) + .1, float(i % 2) + .2])
         for i in range(n)],
        index=index, column="price")
    return SDataFrame([prices])


def test_sklearn_cross_val_score_can_select_rows():
    frame = _frame()
    y = np.asarray([i % 2 for i in range(len(frame))])
    model = make_pipeline(activeTrans(), LogisticRegression(max_iter=200))
    scores = cross_val_score(model, frame, y, cv=3, scoring="accuracy")
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_sklearn_cross_val_score_with_ordered_gap_splits():
    frame = _frame()
    y = np.asarray([i % 2 for i in range(len(frame))])
    model = make_pipeline(activeTrans(), LogisticRegression(max_iter=200))
    scores = cross_val_score(
        model, frame, y,
        cv=PurgedWalkForwardSplit(n_splits=3, gap=2),
        scoring="accuracy")
    assert scores.shape == (3,)


def test_frame_take_preserves_schema_and_does_not_alias_data():
    original = _frame()
    selected = original.take([5, 1, 3], axis=0)
    assert isinstance(selected, SDataFrame)
    assert selected.index.tolist() == original.index[[5, 1, 3]].tolist()
    selected.data[0].values[0].iloc[0] = -123.0
    assert original.data[0].values[5].iloc[0] != -123.0


def test_frame_iloc_selection_and_legacy_two_argument_access():
    frame = _frame()
    assert frame.iloc(3, 0).iloc[0] == 1.0
    assert frame.iloc[3, 0].iloc[0] == 1.0
    subset = frame.iloc[2:5]
    assert isinstance(subset, SDataFrame)
    assert len(subset) == 3
    assert subset.index.tolist() == frame.index[2:5].tolist()


def test_schema_drift_fails_even_when_column_name_matches():
    train = SDataFrame([SData([[1, 2, 3], [2, 4, 6]], column="signal")])
    test = SDataFrame([SData([[3, 4, 5]], dtype="Bag", column="signal")])
    train.fit()
    with pytest.raises(ValueError, match="dtype|type|schema"):
        train.transform(test)


def test_dataframe_fit_checks_indexed_targets_not_only_lengths():
    frame = _frame(8)
    target = pd.Series(range(8), index=frame.index[::-1])
    with pytest.raises(ValueError, match="index|align"):
        frame.fit(y=target)


def test_fitted_frame_and_wrapper_expose_feature_names():
    frame = _frame()
    fitted = frame.fit()
    expected = [f"price__{name}" for name in
                frame.data[0].transformer.fit(frame.data[0]).get_feature_names_out()]
    assert fitted.get_feature_names_out().tolist() == expected
    wrapper = activeTrans().fit(frame)
    assert wrapper.get_feature_names_out().tolist() == expected


def test_take_supports_column_subselection():
    frame = _frame()
    second = SData([[3, 4]] * len(frame), column="other", index=frame.index)
    mixed = frame.join(second)
    selected = mixed.take([1], axis=1)
    assert selected.columns == ["other"]
    assert selected.iloc(0, 0).tolist() == [3, 4]
