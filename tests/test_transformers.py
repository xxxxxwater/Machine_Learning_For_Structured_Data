import numpy as np
import pandas as pd
import pytest

from MLSD import BasicBag, BasicImage, BasicSeries, BasicText, SData, SDataFrame


def test_series_names_remain_identical_between_datasets():
    train = SData([[1, 2, 3, 4], [2, 5, 1, 4]], index=["a", "b"])
    test = SData([[7, 8], [8, 8, 8, 8, 8]], index=["c", "d"])
    trans = BasicSeries().fit(train)
    before = trans.transform(train)
    after = trans.transform(test)
    assert before.shape == (2, 21)
    assert after.shape == (2, 21)
    assert before.columns.tolist() == after.columns.tolist()
    assert after.index.tolist() == ["c", "d"]
    assert np.isnan(after.loc["c", "diff2_min"])
    assert after.loc["d", "value_std"] == 0.0


def test_bag_unordered_and_mesh_features():
    data = SData([[1, 5, 3, 2], [4, 4, 4]], dtype="Bag")
    ordinary = BasicBag().fit_transform(data)
    mesh = BasicBag(mesh=True).fit_transform(data)
    assert ordinary.shape == (2, 7)
    assert mesh.shape == (2, 21)
    assert ordinary.loc[0, "value_median"] == 2.5
    assert mesh.loc[0, "diff1_min"] == -2.0


def test_text_has_stable_training_only_vocabulary():
    train = SData(["bear bear cat", "cat dog"], dtype="Text", index=["t1", "t2"])
    test = SData(["unseen newtoken cat"], dtype="Text", index=["t3"])
    trans = BasicText().fit(train)
    first = trans.transform(train)
    second = trans.transform(test)
    assert first.columns.tolist() == ["bear", "cat", "dog"]
    assert second.columns.tolist() == first.columns.tolist()
    assert second.index.tolist() == ["t3"]
    assert second.shape == (1, 3)
    assert second["cat"].iloc[0] > 0


def test_image_summary_handles_different_sizes_and_rejects_channel_drift():
    images = SData([np.ones((2, 2)), np.arange(16).reshape(4, 4)], dtype="Image")
    trans = BasicImage().fit(images)
    result = trans.transform(images)
    assert result.shape == (2, 7)
    assert result.loc[0, "c0_mean"] == 1.0
    assert result.loc[1, "height"] == 4.0
    with pytest.raises(ValueError, match="channel"):
        trans.transform(SData([np.ones((2, 2, 3))], dtype="Image"))


def test_mixed_frame_keeps_all_features_with_unique_column_names():
    series = SData([[1, 2, 3], [3, 5, 7]], column="price")
    text = SData(["good price", "bad price"], dtype="Text", column="news")
    images = SData([np.ones((2, 2)), np.ones((3, 3))], dtype="Image", column="chart")
    frame = SDataFrame([series, text, images])
    features = frame.fit_transform()
    assert features.shape[0] == 2
    assert features.columns.is_unique
    assert "price__value_mean" in features
    assert "news__price" in features
    assert "chart__height" in features


def test_fit_predict_split_does_not_refit_text_on_test():
    training = SDataFrame([SData(["alpha beta", "beta gamma"],
                                  dtype="Text", column="text")])
    holdout = SDataFrame([SData(["omega alpha"], dtype="Text", column="text",
                                 index=["holdout"])])
    training.fit()
    features = training.transform(holdout)
    assert "text__omega" not in features
    assert features.index.tolist() == ["holdout"]
    with pytest.raises(ValueError, match="Feature columns differ"):
        training.transform(SDataFrame([SData(["alpha"], dtype="Text", column="wrong")]))
