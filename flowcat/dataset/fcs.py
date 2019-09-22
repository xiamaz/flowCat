"""
Basic FCS data types in order to include more metadata.
"""
from __future__ import annotations
from typing import Union, List
import functools
import logging
from collections import namedtuple

import numpy as np

from fcsparser.api import FCSParser
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


def create_meta_from_data(data: np.array, channels: list) -> dict:
    """Get min max ranges from numpy array."""
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    channel_metas = {
        name: ChannelMeta(min_value, max_value, (0, 0), 0)
        for name, min_value, max_value in zip(channels, min_values, max_values)
    }
    return channel_metas


def create_meta_from_fcs(meta: dict, channels: list) -> dict:
    """Get ranges from pnr in metadata."""
    channel_metas = {
        c: ChannelMeta(
            0, int(meta[f"$P{i + 1}R"]),
            tuple(map(float, meta[f"$P{i + 1}E"].split(","))),
            float(meta[f"$P{i + 1}G"]),
        )
        for i, c in enumerate(channels)
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
    # outer join our data
    if channels is None:
        channels = list(set(c for d in fcs_data for c in d.channels))

    aligned = [data.align(channels) for data in fcs_data]
    data = np.concatenate([a.data for a in aligned])
    mask = np.concatenate([a.mask for a in aligned])
    new_data = FCSData((data, mask), channels=channels)
    return new_data


ChannelMeta = namedtuple("ChannelMeta", field_names=["min", "max", "pne", "png"])


class FCSData:
    """Wrap FCS data with additional metadata"""

    __slots__ = (
        "data",  # pd dataframe of data
        "mask",  # np array mask for data
        "meta",  # dict of channel name to channel meta namedtuples
        "channels",  # channel names
    )

    default_encoding = "latin-1"
    default_dataset = 0

    def __init__(
            self,
            initdata: Union[URLPath, FCSData, tuple],
            meta: dict = None,
            channels: list = None,):
        """Create a new FCS object.

        Args:
            initdata: Either tuple of meta and data from fcsparser, string filepath or another FCSData object.
            meta: Dict of fcsmeta named tuples.
        Returns:
            FCSData object.
        """
        if isinstance(initdata, self.__class__):
            if meta is not None:
                raise NotImplementedError("Meta not used for fcs object data input.")

            self.data = initdata.data.copy()
            self.mask = initdata.mask.copy()
            self.meta = initdata.meta.copy()
            self.channels = initdata.channels.copy()
        elif isinstance(initdata, URLPath):
            if meta is not None:
                raise NotImplementedError("Meta not used for urlpath data input.")

            parser = FCSParser(str(initdata), data_set=self.default_dataset, encoding=self.default_encoding)
            self.data = parser.data
            self.mask = np.ones(self.data.shape)
            self.meta = create_meta_from_fcs(parser.annotation, parser.channel_names_s)
            self.channels = list(parser.channel_names_s)
        elif isinstance(initdata, tuple):
            self.data, self.mask = initdata
            if channels is None:
                raise ValueError("Channels needed when initializing from np data")
            self.meta = meta or create_meta_from_data(self.data, channels)
            self.channels = channels
        else:
            raise RuntimeError(
                "Invalid data for FCS. Either Path, similar object or tuple of data and metadata needed.")

        self.data = self.data.astype("float32", copy=False)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ranges_array(self):
        """Get min max ranges as numpy array."""
        return np.array([
            [meta.min, meta.max] for meta in self.meta.values()
        ]).T

    def update_range(self, range_array):
        for name, col in zip(self.meta, range_array.T):
            self.meta[name] = self.meta[name]._replace(min=col[0], max=col[1])

    def update_range_from_dict(self, range_dict):
        for name, (min_val, max_val) in range_dict.items():
            self.meta[name] = self.meta[name]._replace(min=min_val, max=max_val)

    def rename(self, mapping: dict):
        """Rename columns based on the given mapping."""
        self.channels = [mapping.get(name, name) for name in self.channels]
        self.meta = {mapping.get(name, name): data for name, data in self.meta.items()}

    def marker_to_name_only(self) -> FCSData:
        mapping = {c: extract_name(c) for c in self.channels}
        self.rename(mapping)
        return self

    def reorder_channels(self, channels: List[str]) -> FCSData:
        """Reorder columns based on list of channels given."""
        if any(map(lambda c: c not in self.channels, channels)):
            raise ValueError("Some given channels not contained in data.")
        index = np.array([self.channels.index(c) for c in channels])
        self.data = self.data[:, index]
        self.mask = self.mask[:, index]
        self.channels = channels

    def add_missing_channels(self, channels: List[str]) -> FCSData:
        """Add missing columns in the given channel list to the dataframe and
        set them to the missing value."""
        if any(map(lambda c: c in self.channels, channels)):
            raise ValueError("Given channel already in data.")
        cur_channels = self.channels
        new_channels = cur_channels + channels

        cur_dim_a, cur_dim_b = self.data.shape
        new_len = len(channels)
        newdata = np.zeros((cur_dim_a, cur_dim_b + new_len))
        newdata[:, :-new_len] = self.data
        newmask = np.zeros((cur_dim_a, cur_dim_b + new_len))
        newmask[:, :-new_len] = self.mask

        self.meta.update({name: ChannelMeta(0, 0, (0, 0), 0) for name in channels})
        self.data = newdata
        self.mask = newmask
        self.channels = new_channels
        return self

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

        dropped_channels = [c for c in copy.channels if c not in channels]
        if dropped_channels:
            copy.drop_channels(dropped_channels)

        missing_channels = [c for c in channels if c not in copy.channels]
        if missing_channels:
            copy.add_missing_channels(missing_channels)

        copy.reorder_channels(channels)
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
        remaining = [c for c in self.channels if c not in channels]
        remaining_indices = np.array([self.channels.index(c) for c in remaining])
        self.data = self.data[:, remaining_indices]
        self.mask = self.mask[:, remaining_indices]
        self.channels = remaining
        for channel in channels:
            del self.meta[channel]
        return self

    def __repr__(self):
        """Print string representation of the input file."""
        nevents, nchannels = self.data.shape
        return f"<FCS :: {nevents} events :: {nchannels} channels>"
