# pylint: skip-file
# flake8: noqa
from typing import List, Union
import re
import logging
from dataclasses import dataclass, field

from dataslots import with_slots
import numpy as np
import pandas as pd

from flowcat import utils, mappings


LOGGER = logging.getLogger(__name__)


@with_slots
@dataclass
class SOM:
    """Holds self organizing map data with associated metadata."""
    data: Union[np.array, utils.URLPath, str]
    markers: List[str]

    def __post_init__(self):
        if isinstance(self.data, (utils.URLPath, str)):
            self.data = np.load(str(self.data))

    @property
    def dims(self):
        return self.data.shape

    def get_dataframe(self):
        """Return as new pandas dataframe."""
        return pd.DataFrame(self.data.reshape((-1, len(self.markers))), columns=self.markers)

    def get_padded(self, pad_width):
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

    def __repr__(self):
        return f"<SOM {'x'.join(map(str, self.dims))}>"
