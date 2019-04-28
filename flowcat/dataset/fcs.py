import functools
import numpy as np
import pandas as pd

from sklearn import preprocessing, base

import fcsparser


class FCSData:
    """Wrap FCS data with additional metadata"""

    __slots__ = (
        "_meta", "data", "ranges"
    )

    default_encoding = "latin-1"
    default_dataset = 0

    def __init__(self, initdata):
        """Create a new FCS object.

        Args:
            initdata: Either tuple of meta and data from fcsparser, string filepath or another FCSData object.
        Returns:
            FCSData object.
        """
        if isinstance(initdata, self.__class__):
            self._meta = initdata.meta.copy()
            self.ranges = initdata.ranges.copy()
            self.data = initdata.data.copy()
        else:
            # unpack metadata, data tuple
            if isinstance(initdata, tuple):
                meta, data = initdata
            # load using filepath
            else:
                meta, data = fcsparser.parse(
                    str(initdata), data_set=self.default_dataset, encoding=self.default_encoding)
            self._meta = meta
            self.data = data
            self.ranges = self._get_ranges_from_pnr(self._meta)

        self.data = self.data.astype("float64", copy=False)
        self.ranges = self.ranges.astype("float64", copy=False)

    @property
    def channels(self):
        return list(self.data.columns)

    @property
    def meta(self):
        return self._meta

    def align(self, channels, missing_val=-1):
        """Return aligned copy of FCS data."""
        meta = self._meta
        data = self.data
        data = data.assign(**{k: -1 for k in channels if k not in data.columns})
        data = data[channels]
        return self.__class__((meta, data))

    def copy(self):
        return self.__class__(self)

    def drop_empty(self):
        """Drop all channels containing nix in the channel name.
        """
        nix_cols = [c for c in self.data.columns if "nix" in c]
        self.drop_channels(nix_cols)
        return self

    def drop_channels(self, channels):
        """Drop the given columns from the data.
        Args:
            channels: List of channels or channel name to drop. Will not throw an error if the name is not found.
        Returns:
            self. This operation is done in place, so the original object will be modified!
        """
        self.data.drop(channels, axis=1, inplace=True, errors="ignore")
        self.ranges.drop(channels, axis=1, inplace=True, errors="ignore")
        return self

    def normalize(self, scaler=None, fitted=False):
        if scaler is None:
            scaler = FCSStandardScaler()
        return scaler.transform(self) if fitted else scaler.fit_transform(self)

    def scale(self, scaler=None, fitted=False):
        if scaler is None:
            scaler = FCSMinMaxScaler()
        return scaler.transform(self) if fitted else scaler.fit_transform(self)

    def _get_ranges_from_pnr(self, metadata):
        """Get ranges from metainformation."""
        pnr = {
            c: {
                "min": 0,
                "max": int(metadata[f"$P{i + 1}R"])
            } for i, c in enumerate(self.data.columns)
        }
        pnr = pd.DataFrame.from_dict(pnr, orient="columns", dtype="float32")
        return pnr

    def __repr__(self):
        """Print string representation of the input file."""
        nevents, nchannels = self.data.shape
        return f"<FCS :: {nevents} events :: {nchannels} channels>"


class FCSMarkersTransform(base.TransformerMixin, base.BaseEstimator):
    """Filter FCS files based on a list of markers.

    This will drop markers not in the initial list and reorder the columns
    to reflect the sequence in the original markers.

    Missing columns will be instantiated to a default value.
    """

    missing_value = 0.0

    def __init__(self, markers):
        self._markers = markers

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        if isinstance(X, FCSData):
            data = X.data
        else:
            data = X

        missing = [marker for marker in self._markers if marker not in data]
        data[missing] = self.missing_value
        data = data.loc[:, self._markers]

        if isinstance(X, FCSData):
            X.data = data
            X.ranges = X.ranges.loc[:, self._markers]
            X.ranges[missing] = self.missing_value
        return X


class FCSLogTransform(base.BaseEstimator, base.TransformerMixin):
    """Transform FCS files logarithmically.  Currently this does not work
    correctly, since FCS files are not $PnE transformed on import"""

    def transform(self, X, *_):
        names = [n for n in X.columns if "LIN" not in n]
        X[names] = np.log1p(X[names])
        return X

    def fit(self, *_):
        return self


class FCSScatterFilter(base.BaseEstimator, base.TransformerMixin):
    """Remove events with values below threshold in specified channels."""

    def __init__(self, filters=None):
        if filters is None:
            filters = [("SS INT LIN", 0), ("FS INT LIN", 0)]
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


class FCSMinMaxScaler(base.TransformerMixin, base.BaseEstimator):
    """MinMaxScaling with adaptations for FCSData."""

    def __init__(self):
        self._model = None

    def fit(self, X, *_):
        """Fit min max range to the given data."""
        self._model = preprocessing.MinMaxScaler()
        if isinstance(X, FCSData):
            data = X.ranges
        else:
            data = X.data
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        """Transform data to be 0 min and 1 max using the fitted values."""
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


class FCSStandardScaler(base.TransformerMixin, base.BaseEstimator):
    """Standard deviation scaling adapted for FCSData objects."""
    def __init__(self):
        self._model = None

    def fit(self, X, *_):
        """Fit standard deviation to the given data."""
        self._model = preprocessing.StandardScaler()
        if isinstance(X, FCSData):
            data = X.data
        else:
            data = X
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        """Transform data to be zero mean and unit standard deviation"""
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
