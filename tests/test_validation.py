import numpy as np
import pandas as pd
import pytest

from MLSD.validation import audit_features, require_numeric_features


def test_audit_separates_missing_and_infinite_and_constants():
    features = pd.DataFrame({
        "a": [2.0, 2.0, np.nan],
        "b": [1.0, np.inf, 3.0],
        "c": ["x", "y", "z"],
    })
    report = audit_features(features)
    assert report.n_rows == 3
    assert report.n_columns == 3
    assert report.missing == {"a": 1}
    assert report.infinite == {"b": 1}
    assert report.constant_columns == ("a",)
    assert report.non_numeric_columns == ("c",)
    assert not report.is_finite_numeric


def test_sparse_matrix_is_supported():
    sparse = pd.DataFrame.sparse.from_spmatrix(
        __import__("scipy").sparse.csr_matrix([[0.0, 1.0], [0.0, 2.0]]),
        columns=["zero", "value"])
    report = require_numeric_features(sparse, allow_missing=False)
    assert report.constant_columns == ("zero",)
    assert report.is_finite_numeric


def test_guards_reject_problematic_input():
    with pytest.raises(TypeError, match="DataFrame"):
        audit_features([[1.0]])
    with pytest.raises(ValueError, match="empty"):
        audit_features(pd.DataFrame())
    with pytest.raises(ValueError, match="unique"):
        audit_features(pd.DataFrame([[1, 2]], columns=["same", "same"]))
    with pytest.raises(ValueError, match="Infinite"):
        require_numeric_features(pd.DataFrame({"x": [np.inf]}))
    with pytest.raises(ValueError, match="Missing"):
        require_numeric_features(pd.DataFrame({"x": [np.nan]}),
                                 allow_missing=False)
    with pytest.raises(ValueError, match="Nonnumeric"):
        require_numeric_features(pd.DataFrame({"x": ["text"]}))
