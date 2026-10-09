"""Text transformer with train-only vocabulary and sparse TF-IDF output."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from .Series_Transformers import _as_rows


class BasicText(BaseEstimator, TransformerMixin):
    """TF-IDF without vocabulary leakage from the evaluation dataset."""

    def __init__(self, Dreduction=None, lowercase=True, ngram_range=(1, 1),
                 max_features=None, min_df=1, stop_words=None):
        self.Dreduction = Dreduction
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.stop_words = stop_words

    def fit(self, X, y=None):
        if self.Dreduction is not None:
            raise ValueError("BasicText does not support Dreduction; use a pipeline")
        rows, _ = _as_rows(X)
        if not all(isinstance(row, str) for row in rows):
            raise TypeError("BasicText requires strings, not token lists or numbers")
        self.vectorizer_ = TfidfVectorizer(
            lowercase=self.lowercase, ngram_range=self.ngram_range,
            max_features=self.max_features, min_df=self.min_df,
            stop_words=self.stop_words)
        self.vectorizer_.fit(rows)
        return self

    def transform(self, X):
        check_is_fitted(self, "vectorizer_")
        rows, index = _as_rows(X)
        matrix = self.vectorizer_.transform(rows)
        return pd.DataFrame.sparse.from_spmatrix(
            matrix, index=index, columns=self.get_feature_names_out())

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "vectorizer_")
        return self.vectorizer_.get_feature_names_out()
