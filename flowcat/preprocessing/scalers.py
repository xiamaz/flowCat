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
            data = X.ranges_dataframe
        else:
            data = X.data
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        """Transform data to be 0 min and 1 max using the fitted values."""
        orig_data = X.data
        data = self._model.transform(orig_data)
        data = pd.DataFrame(data, columns=orig_data.columns, index=orig_data.index, dtype="float32")

        orig_ranges = X.ranges_dataframe
        ranges = self._model.transform(orig_ranges)
        ranges = pd.DataFrame(ranges, columns=orig_ranges.columns, index=orig_ranges.index, dtype="float32")

        X.set_data(data)
        X.set_ranges_from_dataframe(ranges)
        return X


class FCSStandardScaler(FCSDataMixin, TransformerMixin, BaseEstimator):
    """Standard deviation scaling adapted for FCSData objects."""

    def __init__(self):
        self._model = None

    def fit(self, X, *_):
        """Fit standard deviation to the given data."""
        self._model = StandardScaler()
        data = X.data
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        """Transform data to be zero mean and unit standard deviation"""
        orig_data = X.data
        data = self._model.transform(orig_data)
        data = pd.DataFrame(data, columns=orig_data.columns, index=orig_data.index, dtype="float32")

        orig_ranges = X.ranges_dataframe
        ranges = self._model.transform(orig_ranges)
        ranges = pd.DataFrame(ranges, columns=orig_ranges.columns, index=orig_ranges.index, dtype="float32")

        X.set_data(data)
        X.set_ranges_from_dataframe(ranges)
        return X
