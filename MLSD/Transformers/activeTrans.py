"""Backward-compatible scikit-learn wrapper for structured data transforms."""
from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from ..StructureDataFrame import FrameSamples, SDataFrame


class activeTrans(BaseEstimator, TransformerMixin):
    """Use independent fitted estimators for SData or an entire SDataFrame."""

    def __init__(self, ifSData=False, Trans_in=True, Multi_col=False,
                 New_Trans=None, Reset_default=False):
        self.ifSData = ifSData
        self.Trans_in = Trans_in
        self.Multi_col = Multi_col
        self.New_Trans = New_Trans
        self.Reset_default = Reset_default

    def fit(self, X, y=None):
        if isinstance(X, FrameSamples):
            X = X.frame
        if self.ifSData:
            source = (X.transformer if self.New_Trans is None else self.New_Trans)
            if source is None:
                raise ValueError("SData has no transformer")
            try:
                self.transformer_ = clone(source)
            except TypeError:
                self.transformer_ = deepcopy(source)
            self.transformer_.fit(X, y)
        else:
            if not isinstance(X, SDataFrame):
                raise TypeError("activeTrans(ifSData=False) requires SDataFrame")
            self.frame_ = deepcopy(X)
            if self.New_Trans is not None:
                self.frame_.transformers = [deepcopy(self.New_Trans)
                                            for _ in self.frame_.data]
            self.frame_.fit(y=y)
        return self

    def transform(self, X):
        if isinstance(X, FrameSamples):
            X = X.frame
        if self.ifSData:
            check_is_fitted(self, "transformer_")
            return self.transformer_.transform(X)
        check_is_fitted(self, "frame_")
        return self.frame_.transform(X)

    def get_feature_names_out(self, input_features=None):
        if self.ifSData:
            check_is_fitted(self, "transformer_")
            return self.transformer_.get_feature_names_out(input_features)
        check_is_fitted(self, "frame_")
        return self.frame_.get_feature_names_out(input_features)
