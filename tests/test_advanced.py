import numpy as np
import pytest

from MLSD import FPCA, SData, SDataFrame, activeTrans
from MLSD.Transformers import BsplineSeries, localRSeries


def test_fpca_fit_and_transform_for_variable_length_sequences():
    train = SData([np.sin(np.linspace(0, 3, 20) + k)
                   for k in (0, .1, .3, .7, 1.1)])
    test = SData([np.sin(np.linspace(0, 3, 30)),
                  np.cos(np.linspace(0, 3, 16))])
    model = FPCA(n_components=2, n_points=32).fit(train)
    output = model.transform(test)
    assert output.shape == (2, 2)
    assert output.columns.tolist() == ["fpca_0", "fpca_1"]
    assert np.isfinite(output.to_numpy()).all()


def test_fpca_rejects_short_series():
    with pytest.raises(ValueError, match="two finite"):
        FPCA().fit(SData([[1], [2]]))


def test_active_trans_handles_every_column():
    a = SData([[1, 2, 3], [3, 4, 5]], column="first")
    b = SData([[5, 5, 5], [2, 3, 4]], column="second")
    frame = SDataFrame([a, b])
    pipeline = activeTrans().fit(frame)
    transformed = pipeline.transform(frame)
    assert transformed.shape == (2, 42)
    assert "first__value_min" in transformed
    assert "second__value_min" in transformed


def test_active_trans_single_column_transform():
    data = SData([[1, 2, 3], [4, 5, 6]])
    transformer = activeTrans(ifSData=True)
    output = transformer.fit_transform(data)
    assert output.shape == (2, 21)


def test_optional_bspline_backend():
    pytest.importorskip("patsy")
    data = SData([np.linspace(0, 1, 30), np.linspace(0, 2, 30)])
    features = BsplineSeries(degrees=3, knots=6).fit_transform(data)
    assert features.shape == (2, 6)
    assert np.isfinite(features.to_numpy()).all()


def test_optional_lowess_backend():
    pytest.importorskip("statsmodels")
    data = SData([np.sin(np.linspace(0, 5, 20)),
                  np.cos(np.linspace(0, 5, 20))])
    features = localRSeries(fraction=.5).fit_transform(data)
    assert features.shape == (2, 7)
    assert np.isfinite(features.to_numpy()).all()
