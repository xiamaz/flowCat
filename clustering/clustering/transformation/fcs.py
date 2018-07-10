"""
FCS-level transformations, such as scale transforms or simple filtering
operations.
"""
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


LOGGER = logging.getLogger(__name__)


class MarkersTransform(TransformerMixin, BaseEstimator):
    def __init__(self, markers):
        self._markers = markers

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        return X[self._markers]


class FCSLogTransform(BaseEstimator, TransformerMixin):
    """Transform FCS files logarithmically.  Currently this does not work
    correctly, since FCS files are not $PnE transformed on import"""

    def transform(self, X, *_):
        names = [n for n in X.columns if "LIN" not in n]
        X[names] = np.log1p(X[names])
        return X

    def fit(self, *_):
        return self


class ScatterFilter(BaseEstimator, TransformerMixin):
    """Remove events with values below threshold in specified channels."""

    def __init__(
            self,
            filters=[("SS INT LIN", 0), ("FS INT LIN", 0)],
    ):
        self._filters = filters

    def transform(self, X, *_):
        for column, threshold in self._filters:
            X = X[X[column] > threshold]
        return X

    def fit(self, *_):
        return self
