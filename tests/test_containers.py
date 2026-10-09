import numpy as np
import pandas as pd
import pytest

from MLSD import SData, SDataFrame


def test_ragged_series_preserves_rows_and_labels():
    data = SData([[1, 2], [3, 4, 5]], index=["a", "b"], column="signal")
    assert len(data) == 2
    assert data.size == [2, 3]
    assert data.index.tolist() == ["a", "b"]
    np.testing.assert_allclose(data.mean(), [1.5, 4])


def test_reject_bad_container_dtype_index_and_values():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        SData([[1]], dtype="garbage")
    with pytest.raises(TypeError, match="sequence"):
        SData(123)
    with pytest.raises(ValueError, match="index length"):
        SData([[1]], index=["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        SData([[1], [2]], index=["x", "x"])


def test_missing_value_fill_and_shift():
    data = SData([pd.Series([1, np.nan, 3]), pd.Series([np.nan, 4])])
    data.ffill()
    assert data[0].iloc[1] == 1
    data.bfill()
    assert data[1].iloc[0] == 4
    data.shift(1)
    assert len(data) == 1
    assert data.index.tolist() == [1]


def test_resample_is_not_a_broken_classmethod():
    index = pd.date_range("2024-01-01", periods=4, freq="h")
    data = SData([pd.Series([1, 3, 5, 7], index=index)], index=["asset"])
    sampled = data.C_resample("2h", "mean")
    assert data[0].size == 4
    assert sampled[0].tolist() == [2, 6]
    with pytest.raises(TypeError, match="DatetimeIndex"):
        SData([[1, 2]]).resample("2h")


def test_split_does_not_modify_global_rng_and_preserves_labels():
    data = SData([[i, i + 1] for i in range(20)], index=[f"r{i}" for i in range(20)])
    y = np.arange(20) * 10
    a, ya, b, yb = data.split_train_test(y, random_state=7)
    a2, ya2, b2, yb2 = data.split_train_test(y, random_state=7)
    assert a.index.tolist() == a2.index.tolist()
    np.testing.assert_array_equal(ya, ya2)
    np.testing.assert_array_equal(yb, yb2)
    assert len(set(a.index) & set(b.index)) == 0
    assert sorted(np.r_[ya, yb].tolist()) == sorted(y.tolist())


def test_dataframe_rejects_alignment_and_name_errors():
    a = SData([[1], [2]], index=["a", "b"])
    b = SData([[3], [4]], index=["b", "a"])
    with pytest.raises(ValueError, match="identical ordered indexes"):
        SDataFrame([a, b])
    with pytest.raises(ValueError, match="unique"):
        SDataFrame([a, a], columns=["x", "x"])


def test_dataframe_basics_and_join():
    a = SData([[1, 2], [3, 4]], column="a")
    b = SData([[5, 6], [7, 8]], column="b")
    frame = SDataFrame([a])
    combined = frame.join(b)
    assert len(combined) == 2
    assert combined.columns == ["a", "b"]
    assert combined.iloc(1, 1).tolist() == [7, 8]
    assert combined[0][0].tolist() == [1, 2]
    assert combined.mean().loc[1, "a"] == 3.5
    assert len(frame.columns) == 1
    with pytest.raises(AttributeError):
        _ = combined.not_a_column
