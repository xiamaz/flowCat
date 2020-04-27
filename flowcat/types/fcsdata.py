"""
Basic FCS data types in order to include more metadata.
"""
from typing import Union, List
import functools
import logging
from collections import namedtuple

import numpy as np

from fcsparser.api import FCSParser
import fcsparser

from flowcat.utils import URLPath
from flowcat.types.marker import Marker


LOGGER = logging.getLogger(__name__)


class FCSException(Exception):
    pass


def parse_channel_value(channel_value, data_min, data_max) -> "ChannelMeta":

    data_meta = ChannelMeta(data_min, data_max, (0, 0), 0)

    if isinstance(channel_value, str):
        marker = Marker.name_to_marker(channel_value, meta=data_meta)
    elif channel_value.meta is None:
        marker = channel_value.set_meta(data_meta)
    else:  # marker with valid metadata, return as-is
        marker = channel_value

    return marker

def create_meta_from_data(data: np.array, channels: list) -> dict:
    """Get min max ranges from numpy array."""
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    channel_metas = [
        parse_channel_value(name, min_value, max_value)
        for name, min_value, max_value in zip(channels, min_values, max_values)
    ]
    return channel_metas


def create_meta_from_fcs(meta: dict, channels: list) -> dict:
    """Get ranges from pnr in metadata."""
    def get_gain(i):
        try:
            gain = float(meta[f"$P{i + 1}G"])
        except KeyError:
            LOGGER.debug("No Gain value found for channel %d", i + 1)
            gain = 1.0
        return gain

    return [
        Marker.name_to_marker(c, meta=ChannelMeta(
            0, int(meta[f"$P{i + 1}R"]),
            tuple(map(float, meta[f"$P{i + 1}E"].split(","))),
            get_gain(i)
        ))
        for i, c in enumerate(channels)
    ]


def join_fcs_data(fcs_data: List["FCSData"], channels=None) -> "FCSData":
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


DEFAULT_ENCODING = "latin-1"
DEFAULT_DATASET = 0


class FCSData:
    """Wrap FCS data with additional metadata"""

    __slots__ = (
        "data",  # pd dataframe of data
        "mask",  # np array mask for data
        "channels",  # channel names of type List[Marker]
    )

    def __init__(
            self,
            initdata: Union["URLPath", "FCSData", tuple],
            channels: list = None,):
        """Create a new FCS object.

        Args:
            initdata: Either tuple of meta and data from fcsparser, string filepath or another FCSData object.
            meta: Dict of fcsmeta named tuples.
        Returns:
            FCSData object.
        """
        if isinstance(initdata, self.__class__):
            self.data = initdata.data.copy()
            self.mask = initdata.mask.copy()
            self.channels = initdata.channels.copy()

        elif isinstance(initdata, (URLPath, str)):
            parser = FCSParser(str(initdata), data_set=DEFAULT_DATASET, encoding=DEFAULT_ENCODING)

            self.data = parser.data
            self.mask = np.ones(self.data.shape)
            self.channels = create_meta_from_fcs(parser.annotation, parser.channel_names_s)

        elif isinstance(initdata, tuple):
            self.data, self.mask = initdata

            if channels is None:
                raise ValueError("Channels needed when initializing from np data")

            self.channels = create_meta_from_data(self.data, channels)

        else:
            raise RuntimeError(
                "Invalid data for FCS. Either Path, similar object or tuple of data and metadata needed.")

        self.data = self.data.astype("float32", copy=False)
        self.channels = [Marker.convert(c) for c in self.channels]

    @property
    def markers(self):
        """Synonym added for compatibility with SOM data type."""
        return self.channels

    @property
    def shape(self):
        return self.data.shape

    @property
    def ranges_array(self):
        """Get min max ranges as numpy array."""
        return np.array([
            [m.meta.min, m.meta.max] for m in self.channels
        ]).T

    def update_range(self, range_array):
        updated_channels = [
            channel.set_meta(channel.meta._replace(min=col[0], max=col[1]))
            for channel, col in
            zip(self.channels, range_array.T)
        ]
        self.channels = updated_channels

    def update_range_from_dict(self, range_dict):
        def change_from_dict(channel):
            if str(channel) in range_dict:
                return channel.set_meta(channel.meta._replace(*range_dict[str(channel)]))
            return channel

        updated_channels = [change_from_dict(channel) for channel in self.channels]
        self.channels = updated_channels

    def rename(self, mapping: dict):
        """Rename columns based on the given mapping."""
        def change_channel_name(channel):
            new_name = mapping.get(str(channel))
            if new_name is not None:
                return channel.set_name(new_name)
            return channel
        self.channels = [change_channel_name(c) for c in self.channels]

    def marker_to_name_only(self) -> "FCSData":
        self.channels = [m.set_color(None) for m in self.channels]
        return self

    def reorder_channels(self, channels: List[str]) -> "FCSData":
        """Reorder columns based on list of channels given."""
        if any(map(lambda c: c not in self.channels, channels)):
            raise ValueError("Some given channels not contained in data.")
        return self[channels]

    def add_missing_channels(self, channels: List[str]) -> "FCSData":
        """Add missing columns in the given channel list to the dataframe and
        set them to the missing value."""
        if any(map(lambda c: c in self.channels, channels)):
            raise ValueError("Given channel already in data.")
        channels = [Marker.convert(c).set_meta(ChannelMeta(0, 0, (0, 0), 0)) for c in channels]
        cur_channels = self.channels
        new_channels = cur_channels + channels

        cur_dim_a, cur_dim_b = self.data.shape
        new_len = len(channels)
        newdata = np.zeros((cur_dim_a, cur_dim_b + new_len))
        newdata[:, :-new_len] = self.data
        newmask = np.zeros((cur_dim_a, cur_dim_b + new_len))
        newmask[:, :-new_len] = self.mask

        self.data = newdata
        self.mask = newmask
        self.channels = new_channels
        return self

    def align(
            self,
            channels: List[str],
            name_only: bool = False,
            inplace: bool = False) -> "FCSData":
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
            copy = copy.marker_to_name_only()

        dropped_channels = [c for c in copy.channels if c not in channels]
        if dropped_channels:
            copy = copy.drop_channels(dropped_channels)

        missing_channels = [c for c in channels if c not in copy.channels]
        if missing_channels:
            copy = copy.add_missing_channels(missing_channels)

        copy = copy.reorder_channels(channels)
        return copy

    def copy(self) -> "FCSData":
        return self.__class__(self)

    def drop_empty(self) -> "FCSData":
        """Drop all channels containing nix in the channel name.
        """
        nix_cols = [c for c in self.channels if c.antibody == "nix"]
        return self.drop_channels(nix_cols)

    def drop_channels(self, channels) -> "FCSData":
        """Drop the given columns from the data.
        Args:
            channels: List of channels or channel name to drop. Will not throw an error if the name is not found.
        Returns:
            FCSData object with channels removed.
        """
        remaining = [c for c in self.channels if c not in channels]
        return self[remaining]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            ridx, cidx = idx
        else:
            ridx, cidx = slice(None), idx
        cidx = [self.markers.index(i) for i in cidx]
        sel_data = self.data[ridx, cidx]
        sel_mask = self.mask[ridx, cidx]
        sel_channels = [self.channels[i] for i in cidx]

        return FCSData((sel_data, sel_mask), sel_channels)

    def __repr__(self):
        """Print string representation of the input file."""
        nevents, nchannels = self.data.shape
        return f"<FCS :: {nevents} events :: {nchannels} channels>"
