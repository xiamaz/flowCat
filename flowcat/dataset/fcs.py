"""
Basic FCS data types in order to include more metadata.
"""
from __future__ import annotations
from typing import Union, List
import functools
import logging

import numpy as np
import pandas as pd

from sklearn import preprocessing, base

import fcsparser

from flowcat.utils import URLPath, outer_interval
from flowcat.mappings import MARKER_NAME_MAP


LOGGER = logging.getLogger(__name__)


def extract_name(marker):
    splitted = marker.split("-")
    if len(splitted) == 2:
        name, _ = splitted
    else:
        name = marker

    name = MARKER_NAME_MAP.get(name, name)
    return name


def join_fcs_data(fcs_data: List[FCSData], channels=None) -> FCSData:
    """Join the given fcs files.

    Args:
        fcs_data: List of fcs files.
        channels: Optionally align new fcs data file to the given channels.
    Returns:
        Merged single FCSData containing entries from all given fcsdata files.
    """
    if channels is None:
        channels = fcs_data[0].channels

    dfs = [data.align(channels).data for data in fcs_data]
    joined = pd.concat(dfs)

    all_ranges = [data.ranges for data in fcs_data]
    ranges = functools.reduce(
        lambda x, y: {name: outer_interval(x[name], y[name]) for name in channels},
        all_ranges
    )

    return FCSData(joined, meta=None, ranges=ranges)


class FCSData:
    """Wrap FCS data with additional metadata"""

    __slots__ = (
        "_meta",  # dict of metadata
        "data",  # pd dataframe of data
        "ranges"  # dict of pd intervals
    )

    default_encoding = "latin-1"
    default_dataset = 0

    def __init__(
            self,
            initdata: Union[pd.DataFrame, URLPath, FCSData],
            meta: dict = None,
            ranges: dict = None):
        """Create a new FCS object.

        Args:
            initdata: Either tuple of meta and data from fcsparser, string filepath or another FCSData object.
            meta: Manually provide dict of metadata.
            ranges: Manually provide ranges
        Returns:
            FCSData object.
        """
        if isinstance(initdata, self.__class__):
            self._meta = initdata.meta.copy()
            self.ranges = initdata.ranges.copy()
            self.data = initdata.data.copy()
        elif isinstance(initdata, URLPath):
            parsed_meta, data = fcsparser.parse(
                str(initdata), data_set=self.default_dataset, encoding=self.default_encoding)
            self.data = data
            self._meta = meta or parsed_meta
            self.ranges = ranges or self._get_ranges_from_pnr(self._meta)
        elif isinstance(initdata, pd.DataFrame):
            self.data = initdata
            self._meta = meta
            self.ranges = ranges or self._get_ranges_from_pnr(self._meta)
        else:
            raise RuntimeError(
                "Invalid data for FCS. Either Path, similar object or tuple of data and metadata needed.")

        self.data = self.data.astype("float32", copy=False)

    @property
    def channels(self) -> List[str]:
        return list(self.data.columns)

    @property
    def meta(self) -> dict:
        return self._meta

    def rename(self, mapping: dict):
        """Rename columns based on the given mapping."""
        self.data.rename(mapping, axis=1, inplace=True)
        self.ranges = {mapping.get(name, name) for name, interval in self.ranges.items()}

    def channel_mask(self, channels: List[str]):
        """Return a 1/0 int array for the given channels, whether channels
        exist or not."""
        return np.array([c in self.data.columns for c in channels])

    def align(self, channels: List[str], missing_val=np.nan, name_only: bool = False) -> FCSData:
        """Return aligned copy of FCS data.

        Args:
            channels: List of channels to be aligned to.
            missing_val: Value to be used for missing columns.
            name_only: Only use the first part of the name.

        Returns:
            Aligned copy of the data.
        """
        copy = self.copy()
        if name_only:
            mapping = {c: extract_name(c) for c in copy.channels}
            copy.rename(mapping)

        cur_channels = copy.channels
        missing_cols = {k: missing_val for k in channels if k not in cur_channels}
        if len(missing_cols) / float(len(channels)) > 0.5:
            LOGGER.warning("More %d columns are missing. Maybe something is wrong.", len(missing_cols))

        data = copy.data
        data = data.assign(**missing_cols)
        data = data[channels]
        copy.data = data

        copy.ranges.update(
            {
                name: pd.Interval(missing_val, missing_val)
                for name, missing_val in missing_cols.items()
            }
        )
        return copy

    def copy(self) -> FCSData:
        return self.__class__(self)

    def drop_empty(self) -> FCSData:
        """Drop all channels containing nix in the channel name.
        """
        nix_cols = [c for c in self.data.columns if "nix" in c]
        self.drop_channels(nix_cols)
        return self

    def drop_channels(self, channels) -> FCSData:
        """Drop the given columns from the data.
        Args:
            channels: List of channels or channel name to drop. Will not throw an error if the name is not found.
        Returns:
            self. This operation is done in place, so the original object will be modified!
        """
        self.data.drop(channels, axis=1, inplace=True, errors="ignore")
        for channel in channels:
            del self.ranges[channel]
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
            c: pd.Interval(0, int(metadata[f"$P{i + 1}R"]), closed="both")
            for i, c in enumerate(self.data.columns)
        }
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
