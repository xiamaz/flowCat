import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator


from flowcat.dataset import fcs
from . import FCSDataMixin


class FCSMinMaxScaler(FCSDataMixin, TransformerMixin, BaseEstimator):
    """MinMaxScaling with adaptations for FCSData."""

    def __init__(self, fit_to_range=False):
        self._model = None
        self._fit_to_range = fit_to_range

    def fit(self, X, *_):
        """Fit min max range to the given data."""
        self._model = MinMaxScaler()
        if self._fit_to_range:
            data = X.ranges_array
        else:
            data = X.data
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        """Transform data to be 0 min and 1 max using the fitted values."""
        X.data = self._model.transform(X.data)
        X.update_range(self._model.transform(X.ranges_array))
        return X


class FCSStandardScaler(FCSDataMixin, TransformerMixin, BaseEstimator):
    """Standard deviation scaling adapted for FCSData objects."""

    def __init__(self):
        self._model = None

    def fit(self, X, *_):
        """Fit standard deviation to the given data."""
        self._model = StandardScaler().fit(X.data)
        return self

    def transform(self, X, *_):
        """Transform data to be zero mean and unit standard deviation"""
        X.data = self._model.transform(X.data)
        X.update_range(self._model.transform(X.ranges_array))
        return X
