"""
FCS-level transformations, such as scale transforms or simple filtering
operations.

These will take either pandas dataframes or FCSData objects.
"""
import logging
import functools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import preprocessing

from ..data.fcsdata import FCSData


LOGGER = logging.getLogger(__name__)


class MarkersTransform(TransformerMixin, BaseEstimator):
    def __init__(self, markers):
        self._markers = markers

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        if isinstance(X, FCSData):
            X.data = X.data.loc[:, self._markers]
        else:
            X = X.loc[:, self._markers]
        return X


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
        if isinstance(X, FCSData):
            selected = functools.reduce(
                lambda x, y: x & y, [X.data[c] > t for c, t in self._filters])
            X.data = X.data.loc[selected, :]
        else:
            selected = functools.reduce(
                lambda x, y: x & y, [X[c] > t for c, t in self._filters])
            X = X.loc[selected, :]
        return X

    def fit(self, *_):
        return self


class FCSMinMaxScaler(TransformerMixin, BaseEstimator):
    """MinMaxScaling with adaptations for FCSData."""

    def __init__(self):
        self._model = preprocessing.MinMaxScaler()

    def fit(self, X, *_):
        if isinstance(X, FCSData):
            data = X.ranges
        else:
            data = X.data
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        if isinstance(X, FCSData):
            data = self._model.transform(X.data)
            X.data = pd.DataFrame(data, columns=X.data.columns, index=X.data.index)
            ranges = self._model.transform(X.ranges)
            X.ranges = pd.DataFrame(ranges, columns=X.ranges.columns, index=X.ranges.index)
        elif isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                self._model.transform(X),
                columns=X.columns, index=X.index)
        else:
            X = self._model.transform(X)
        return X


class FCSStandardScaler(TransformerMixin, BaseEstimator):
    """Standard deviation scaling adapted for FCSData objects."""
    def __init__(self):
        self._model = preprocessing.StandardScaler()

    def fit(self, X, *_):
        if isinstance(X, FCSData):
            data = X.data
        else:
            data = X
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        if isinstance(X, FCSData):
            data = self._model.transform(X.data)
            ranges = self._model.transform(X.ranges)
            X.data = pd.DataFrame(data, columns=X.data.columns, index=X.data.index)
            X.ranges = pd.DataFrame(ranges, columns=X.ranges.columns, index=X.ranges.index)
        elif isinstance(X, pd.DataFrame):
            data = self._model.transform(X)
            X = pd.DataFrame(data, columns=X.columns, index=X.index)
        else:
            X = self._model.transform(X)
        return X

