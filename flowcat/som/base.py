from __future__ import annotations
from typing import List, Union
import re
import logging
import dataclasses

import numpy as np
import pandas as pd

from flowcat import utils, configuration, mappings


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class SOM:
    """Holds self organizing map data with associated metadata."""
    data: pd.DataFrame
    path: utils.URLPath = None
    cases: List[str] = dataclasses.field(default_factory=list)
    tube: int = -1
    material: mappings.Material = None
    transforms: List[dict] = dataclasses.field(default_factory=list)

    @classmethod
    def from_path(cls, data_path, config_path, **kwargs):
        data = utils.load_csv(data_path)
        try:
            config = utils.load_json(config_path)
        except FileNotFoundError:
            config = {}

        kwargs = {**config, **kwargs}
        return cls(data, **kwargs)

    @property
    def dims(self):
        rows = self.data.shape[0]
        sq_size = int(np.sqrt(rows))
        return (sq_size, sq_size)

    @property
    def markers(self):
        return self.data.columns.values

    @property
    def config(self):
        return {
            "cases": self.cases,
            "tube": self.tube,
            "transforms": self.transforms,
        }

    def np_array(self, pad_width=0):
        data = np.reshape(self.data.values, (*self.dims, -1))
        if pad_width:
            data = np.pad(data, pad_width=[
                (pad_width, pad_width),
                (pad_width, pad_width),
                (0, 0),
            ], mode="wrap")
        return data

    def __repr__(self):
        return f"<SOM {'x'.join(map(str, self.dims))} Tube:{self.tube}>"


class SOMCollection:
    """Holds multiple SOM, eg for different tubes for a single patient."""

    path = None
    config = None
    cases = None
    tubes = None

    def __init__(self, path=None, tubes=None, tubepaths=None, cases=None, config=None):
        self.path = path
        self.cases = cases or []
        self.tubes = tubes or []
        self._tubepaths = tubepaths or {}
        self.config = config
        self._index = 0
        self._max_index = 0

        self._data = {}

    @classmethod
    def from_path(cls, path, subdirectory, tubes=None, **kwargs):
        path = utils.URLPath(path)
        if tubes:
            tubepaths = {
                tube: get_som_tube_path(path, tube, subdirectory)[0] for tube in tubes
            }
        else:
            if subdirectory:
                paths = path.glob("t*.csv")
            else:
                parent = path.local.parent
                paths = [p for p in parent.glob(f"{path.local.name}*.csv")]

            tubepaths = {
                int(m[1]): p for m, p in
                [(re.search(r"t(\d+)\.csv", str(path)), path) for path in paths]
                if m is not None
            }
        tubes = sorted(tubepaths.keys())
        # load config if exists
        conf_path = path / "config.toml"
        if conf_path.exists():
            config = configuration.SOMConfig.from_file(conf_path)
        else:
            config = None
        return cls(path=path, tubes=tubes, tubepaths=tubepaths, config=config)

    def load(self):
        """Load all tubes into cache."""
        for tube in self.tubes:
            self.get_tube(tube)

    def get_tube(self, tube):
        if tube in self._data:
            return self._data[tube]
        if tube not in self.tubes:
            return None
        path = self._tubepaths[tube]
        data = SOM.from_path(path, str(path)[:-3] + "json", tube=tube, cases=self.cases)
        self._data[tube] = data
        return data

    def add_som(self, data):
        self._data[data.tube] = data
        if data.tube not in self.tubes:
            self.tubes.append(data.tube)

    @property
    def dims(self):
        if self.config:
            m = self.config("tfsom", "m")
            n = self.config("tfsom", "n")
        else:
            data = self.get_tube(self.tubes[0])
            return data.dims
        return (m, n)

    def __iter__(self):
        self._index = self.tubes[0]
        self._max_index = len(self.tubes)
        return self

    def __next__(self):
        if self._index < self._max_index:
            index = self._index
            self._index += 1
            return self.get_tube(index)
        raise StopIteration

    def __repr__(self):
        return f"<SOMCollection: Tubes: {self.tubes} Loaded: {len(self._data)}>"


def get_som_tube_path(
        path: Union[str, utils.URLPath],
        tube: int,
        subdirectory: bool) -> utils.URLPath:
    path = utils.URLPath(path)
    if subdirectory:
        result = path / f"t{tube}"
    else:
        result = path + f"_t{tube}"
    return (result + ".csv", result + ".json")


def load_som(
        path: Union[str, utils.URLPath],
        subdirectory: bool = False,
        tube: Union[int, list] = None) -> Union[SOMCollection, SOM]:
    """Load soms into a som collection or if tube specified into a single SOM."""
    # Load single SOM
    if isinstance(tube, int):
        inpaths = get_som_tube_path(path, tube, subdirectory)
        return SOM.from_path(*inpaths, tube=tube)
    # Load multiple SOM into a SOM collection
    return SOMCollection.from_path(path, tubes=tube, subdirectory=subdirectory)


def save_som(
        som: Union[SOMCollection, SOM],
        path: Union[str, utils.URLPath],
        subdirectory: bool = False,
        save_config: bool = True):
    """Save som object to the given destination.
    Params:
        som: Either a SOM collection or a single SOM object.
        path: Destination path
        subdirectory: Save files to separate files with path as directory name.
    """
    if isinstance(som, SOMCollection):
        soms = som
    elif isinstance(som, SOM):
        soms = [som]
    else:
        raise TypeError

    path = utils.URLPath(path)

    for som_obj in soms:
        data_dest, conf_dest = get_som_tube_path(path, som_obj.tube, subdirectory)
        LOGGER.debug("Saving %s to %s", som_obj, data_dest)
        utils.save_csv(som_obj.data, data_dest)
        if save_config:
            utils.save_json(som_obj.config, conf_dest)
