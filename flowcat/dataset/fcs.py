"""
Basic FCS data types in order to include more metadata.
"""
from __future__ import annotations
from typing import Union, List
import functools
import logging
from collections import namedtuple

import numpy as np
import pandas as pd

import fcsparser

from flowcat.utils import URLPath, outer_interval
from flowcat.mappings import MARKER_NAME_MAP


LOGGER = logging.getLogger(__name__)


class FCSException(Exception):
    pass


def extract_name(marker):
    splitted = marker.split("-")
    if len(splitted) == 2:
        name, _ = splitted
    else:
        name = marker

    name = MARKER_NAME_MAP.get(name, name)
    return name


def create_meta_from_data(data: pd.DataFrame) -> dict:
    """Get min max ranges from pandas dataframe."""
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    channel_metas = {}
    for name, min_value in min_values.items():
        max_value = max_values[name]
        channel_metas[name] = ChannelMeta(
            pd.Interval(min_value, max_value, closed="both"),
            True)
    return channel_metas


def create_meta_from_fcs(meta: dict, data: pd.DataFrame) -> dict:
    """Get ranges from pnr in metadata."""
    channel_metas = {
        c: ChannelMeta(
            pd.Interval(0, int(meta[f"$P{i + 1}R"]), closed="both"),
            True)
        for i, c in enumerate(data.columns)
    }
    return channel_metas


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

    all_meta = [data.meta for data in fcs_data]
    meta = functools.reduce(
        lambda x, y: {
            name: ChannelMeta(
                outer_interval(x[name].range, y[name].range),
                x[name].exists or y[name].exists
            ) for name in channels
        }, all_meta
    )

    return FCSData(joined, meta=meta)


ChannelMeta = namedtuple("ChannelMeta", field_names=["range", "exists"])


class FCSData:
    """Wrap FCS data with additional metadata"""

    __slots__ = (
        "_data",  # pd dataframe of data
        "_meta",  # dict of channel meta namedtuples
        "_channels",  # channel names
    )

    default_encoding = "latin-1"
    default_dataset = 0

    def __init__(
            self,
            initdata: Union[pd.DataFrame, URLPath, FCSData],
            meta: dict = None):
        """Create a new FCS object.

        Args:
            initdata: Either tuple of meta and data from fcsparser, string filepath or another FCSData object.
            meta: Dict of fcsmeta named tuples.
        Returns:
            FCSData object.
        """
        if isinstance(initdata, self.__class__):
            if meta is not None:
                raise NotImplementedError("Meta not used for urlpath fcs objects.")

            self._data = initdata.data.copy()
            self._meta = initdata.meta.copy()
        elif isinstance(initdata, URLPath):
            if meta is not None:
                raise NotImplementedError("Meta not used for urlpath fcs objects.")

            meta, data = fcsparser.parse(
                str(initdata), data_set=self.default_dataset, encoding=self.default_encoding)
            self._data = data
            self._meta = create_meta_from_fcs(meta, data)
        elif isinstance(initdata, pd.DataFrame):
            self._data = initdata
            self._meta = meta or create_meta_from_data(initdata)
        else:
            raise RuntimeError(
                "Invalid data for FCS. Either Path, similar object or tuple of data and metadata needed.")

        self._data = self._data.astype("float32", copy=False)
        self._channels = None

    @property
    def shape(self):
        return self._data.shape

    @property
    def channels(self) -> List[str]:
        if self._channels is None:
            self._channels = list(self._data.columns)
        return self._channels

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def ranges_dataframe(self) -> pd.DataFrame:
        """Return min and max range as dataframe."""
        ranges_dict = {
            n: {"min": m.range.left, "max": m.range.right}
            for n, m in self.meta.items()
        }
        ranges = pd.DataFrame(data=ranges_dict, dtype="float32")
        return ranges

    def set_data(self, data):
        if set(list(data.columns)) != set(self.channels):
            raise FCSException("Channels are different in new data.")
        self._data = data

    def set_ranges_from_dataframe(self, ranges: pd.DataFrame):
        """Set ranges from ranges dataframe with optional info on closedness."""
        ranges = {
            name: pd.Interval(
                min_max["min"], min_max["max"],
                self.meta[name].range.closed
            ) for name, min_max in ranges.to_dict().items()
        }
        self.set_ranges(ranges)

    def set_ranges(self, ranges: dict):
        """Set meta ranges from dict mapping channel names to intervals."""
        for name, channel_range in ranges.items():
            self.set_range(name, channel_range)

    def set_range(self, name, channel_range: pd.Interval):
        self.meta[name] = ChannelMeta(channel_range, self.meta[name].exists)

    def rename(self, mapping: dict):
        """Rename columns based on the given mapping."""
        self._data.rename(mapping, axis=1, inplace=True)
        self._meta = {mapping.get(name, name) for name, meta in self.meta.items()}
        self._channels = None

    def marker_to_name_only(self) -> FCSData:
        mapping = {c: extract_name(c) for c in self.channels}
        self.rename(mapping)
        return self

    def add_missing_columns(self, channels: List[str], missing_val=np.nan) -> FCSData:
        """Add missing columns in the given channel list to the dataframe and
        set them to the missing value."""
        cur_channels = self.channels
        missing_cols = {k: missing_val for k in channels if k not in cur_channels}
        if len(missing_cols) / float(len(channels)) > 0.5:
            LOGGER.warning("More %d columns are missing. Maybe something is wrong.", len(missing_cols))

        self._data = self._data.assign(**missing_cols)
        self._meta.update(
            {
                name: ChannelMeta(
                    pd.Interval(missing_val, missing_val),
                    False)
                for name, missing_val in missing_cols.items()
            }
        )
        self._channels = None
        return self

    def channel_mask(self):
        """Return a 1/0 int array for the given channels, whether channels
        exist or not."""
        return np.array([self.meta[c].exists for c in self.channels])

    def align(
            self,
            channels: List[str],
            missing_val=np.nan,
            name_only: bool = False,
            inplace: bool = False) -> FCSData:
        """Return aligned copy of FCS data.

        Args:
            channels: List of channels to be aligned to.
            missing_val: Value to be used for missing columns.
            name_only: Only use the first part of the name.

        Returns:
            Aligned copy of the data.
        """
        if inplace:
            copy = self
        else:
            copy = self.copy()

        if name_only:
            copy.marker_to_name_only()

        copy.add_missing_columns(channels, missing_val=missing_val)

        copy._data = copy._data[channels]
        copy._meta = {name: copy._meta[name] for name in channels}
        copy._channels = channels
        return copy

    def copy(self) -> FCSData:
        return self.__class__(self)

    def drop_empty(self) -> FCSData:
        """Drop all channels containing nix in the channel name.
        """
        nix_cols = [c for c in self.channels if "nix" in c]
        self.drop_channels(nix_cols)
        return self

    def drop_channels(self, channels) -> FCSData:
        """Drop the given columns from the data.
        Args:
            channels: List of channels or channel name to drop. Will not throw an error if the name is not found.
        Returns:
            self. This operation is done in place, so the original object will be modified!
        """
        self._data.drop(channels, axis=1, inplace=True, errors="ignore")
        for channel in channels:
            del self._meta[channel]
        self._channels = None
        return self

    def __repr__(self):
        """Print string representation of the input file."""
        nevents, nchannels = self.data.shape
        return f"<FCS :: {nevents} events :: {nchannels} channels>"
