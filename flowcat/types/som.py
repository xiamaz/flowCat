# pylint: skip-file
# flake8: noqa
from typing import List, Union
import re
import logging

from dataclasses import dataclass, field
from dataslots import with_slots

import numpy as np
import pandas as pd

from flowcat.utils import URLPath


LOGGER = logging.getLogger(__name__)


@with_slots
@dataclass
class SOM:
    """Holds self organizing map data with associated metadata."""
    data: Union[np.array, URLPath, str]
    markers: List[str]

    def __post_init__(self):
        if isinstance(self.data, (URLPath, str)):
            self.data = np.load(str(self.data))

    @property
    def dims(self) -> tuple:
        return self.data.shape

    @property
    def shape(self) -> tuple:
        """Synonym added for compatibility with FCS data type."""
        return self.dims

    def get_dataframe(self) -> pd.DataFrame:
        """Return as new pandas dataframe."""
        return pd.DataFrame(self.data.reshape((-1, len(self.markers))), columns=self.markers)

    def get_padded(self, pad_width) -> np.array:
        """Return as new numpy array. Optionally with padding by adding zeros
        to the borders of the SOM.

        Args:
            pad_width: Additional padding for SOM on borders. The width is
                       added to each border.
        """
        data = np.reshape(self.data.values, (*self.dims, -1))
        data = np.pad(data, pad_width=[
            (pad_width, pad_width),
            (pad_width, pad_width),
            (0, 0),
        ], mode="wrap")
        return data

    def __getitem__(self, idx) -> "SOM":
        if isinstance(idx, tuple):
            ridx, cidx, midx = idx
        else:
            ridx, cidx, midx = slice(None), slice(None), idx

        midx = [self.markers.index(i) for i in midx]
        sel_data = self.data[ridx, cidx, midx]
        sel_markers = [self.markers[i] for i in midx]
        return SOM(sel_data, sel_markers)

    def __repr__(self) -> str:
        return f"<SOM {'x'.join(map(str, self.dims))}>"
