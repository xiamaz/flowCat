"""
Functions to create SOMs.

These functions use configuration objects, as defined properly in configurations.
"""
import logging
import collections
import time
import re

import numpy as np
import pandas as pd

from . import configuration, utils, mappings
from .dataset import case_dataset
from .models import tfsom


LOGGER = logging.getLogger(__name__)


def get_som_tube_path(path, tube, subdirectory):
    path = utils.URLPath(path)
    if subdirectory:
        result = path / f"t{tube}.csv"
    else:
        result = path + f"_t{tube}.csv"
    return result


def load_som(path, subdirectory=False, tube=None):
    """Load soms into a som collection or if tube specified into a single SOM."""
    # Load single SOM
    if tube:
        inpath = get_som_tube_path(path, tube, subdirectory)
        return SOM.from_path(inpath, tube=tube)
    # Load multiple SOM into a SOM collection
    return SOMCollection.from_path(path, subdirectory=subdirectory)


def save_som(som, path, subdirectory=False):
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

    for som in soms:
        dest = get_som_tube_path(path, som.tube, subdirectory)
        LOGGER.debug("Saving %s to %s", som, dest)
        utils.save_csv(som.data, dest)


class SOM:
    """Holds self organizing map data with associated metadata."""
    data = None
    path = None
    cases = None
    tube = -1

    def __init__(self, data, path=None, cases=None, tube=-1):
        self.data = data
        self.path = path
        self.cases = cases
        self.tube = tube

    @classmethod
    def from_path(cls, path, **kwargs):
        data = utils.load_csv(path)
        return cls(data, **kwargs)

    @property
    def dims(self):
        rows = self.data.shape[0]
        sq_size = int(np.sqrt(rows))
        return (sq_size, sq_size)

    @property
    def markers(self):
        return self.data.columns.values

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
    def from_path(cls, path, subdirectory, **kwargs):
        path = utils.URLPath(path)
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
        data = SOM.from_path(path, tube=tube, cases=self.cases)
        self._data[tube] = data
        return data

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


class ReferenceConfig(configuration.SOMConfig):
    """Basic configuration for reference generation."""

    name = "refconfig"
    desc = "Reference SOM configuration"
    default = ""

    @classmethod
    def generate_config(cls, args):
        """Quickly change some important values."""
        name = args.name
        subsample_size = 8192
        gridsize = 32
        data = {
            "name": name,
            "dataset": {
                "labels": "data/selected_cases.txt",
                "filters": {
                    "num": 1,
                    "groups": None,
                    "counts": subsample_size,
                    "infiltration": None,
                    "infiltration_max": None,
                },
                "preprocessing": "zscore",
                "selected_markers": mappings.CHANNEL_CONFIGS["CLL-9F"],
            },
            "tfsom": {
                "model_name": name,
                "m": gridsize,
                "n": gridsize,
                "map_type": "toroid",
                "max_epochs": 10,
                "batch_size": 1,
                "subsample_size": subsample_size,
                "initial_radius": int(gridsize / 2),
                "initialization_method": "random",
            }
        }
        return cls(data)


class IndivConfig(configuration.SOMConfig):
    """Configuration to generate individual SOMs."""

    name = "indivconfig"
    desc = "Individual som configuration"
    default = ""

    @classmethod
    def generate_config(cls, args):
        """Quickly change some important values."""
        name = args.name
        subsample_size = 8192
        gridsize = 32
        data = {
            "name": name,
            "dataset": {
                "labels": None,
                "filters": {
                    "num": None,
                    "groups": None,
                    "counts": subsample_size,
                    "infiltration": None,
                    "infiltration_max": None,
                },
                "preprocessing": "zscore",
                "selected_markers": mappings.CHANNEL_CONFIGS["CLL-9F"],
            },
            "tfsom": {
                "model_name": name,
                "m": gridsize,
                "n": gridsize,
                "map_type": "toroid",
                "max_epochs": 10,
                "batch_size": 1,
                "subsample_size": subsample_size,
                "initial_radius": int(gridsize / 2),
                "initialization_method": "reference",
            },
            "somnodes": {
                "fitmap_args": {
                    "max_epochs": 10,
                    "initial_learn": 0.5,
                    "end_learn": 0.1,
                    "initial_radius": 16,
                    "end_radius": 1,
                }
            }
        }
        return cls(data)


def get_config_path(path):
    """Get configuration for the given dataset."""
    return path / "config.toml"


def create_datasets(cases, dataset_config):
    """Create dataset generators."""
    filters = dataset_config["filters"]
    tubes = filters["tubes"]
    markers = dataset_config["selected_markers"]
    preprocessing = tfsom.get_generator(dataset_config["preprocessing"])

    dataset = case_dataset.CaseView(cases, selected_markers=markers, selected_tubes=tubes)

    for tube in tubes:
        tubeview = dataset.get_tube(tube)
        yield tube, preprocessing(tubeview)


def create_som(cases, config, tensorboard_path=None, seed=None, reference=None):
    """Create a SOM for the given list of cases with the configuration."""

    markers = config("dataset", "selected_markers")

    somweights = {}
    for tube, (datagen, length) in create_datasets(cases, config("dataset")):
        tmarkers = markers[tube]
        treference = reference[tube] if reference else None
        model = tfsom.TFSom(
            channels=tmarkers, tube=tube, reference=treference,
            **config("tfsom"),
            tensorboard_dir=tensorboard_path,
            seed=seed)
        # train the network
        with utils.timer("Training time"):
            model.train(datagen())

        somweights[tube] = pd.DataFrame(model.output_weights, columns=tmarkers)

    return somweights


def load_som_dict(path, tubes=None, suffix=False):
    """Load SOM data from the given location.

    SOMs are saved to multiple files, each representing data from a different
    tube.

    Args:
        path: Path to get data from.
        tubes: Tubes to be loaded.
        suffix: Enable to append tube as suffix to the path instead of using path as
            a directory level.
    Returns:
        Dictionary mapping tube to SOM data.
    """
    path = utils.URLPath(path)
    data = {}
    if tubes is None:
        paths = path.glob("t*.csv")
        tubes = [
            int(m[1]) for m in
            [re.search(r"t(\d+)\.csv", str(path)) for path in paths]
            if m is not None
        ]
    for tube in tubes:
        tpath = get_som_tube_path(path, tube, not suffix)
        data[tube] = utils.load_csv(tpath)
    return data


def save_som_dict(data, path, suffix=False):
    """Save SOM to the specified location.
    Args:
        data: Dict mapping tubes to SOM data.
        path: Output path to save files.
        suffix: Enable suffix mode will save files to path by appending the tube
            as a suffix to the output path. Otherwise the path will be treated as a
            folder with files saved to t{tube}.csv inside it.
    """
    path = utils.URLPath(path)
    for tube, somdata in data.items():
        outpath = get_som_tube_path(path, tube, not suffix)
        utils.save_csv(somdata, outpath)


def create_filtered_data(config, pathconfig=None):
    """Create filtered dataset."""
    if pathconfig is not None:
        casespath = utils.get_path(config("dataset", "names", "FCS"), pathconfig("input", "FCS"))
    else:
        casespath = config("dataset", "names", "FCS")

    labels = utils.load_labels(config("dataset", "labels"))
    filters = config("dataset", "filters")
    cases = case_dataset.CaseCollection.from_path(casespath)
    data = cases.filter(labels=labels, **filters)

    return data


def create_indiv_soms(data, config, path, tensorboard_dir=None, pathconfig=None, reference=None):
    """Create indiv soms using a generator."""
    path = utils.URLPath(path)
    if reference:
        selected_markers = {k: list(v.columns) for k, v in reference.items()}
        data = data.filter(selected_markers=selected_markers)
    else:
        selected_markers = None

    meta = {"label": [], "randnum": []}
    for tube in data.selected_tubes:
        # get the correct data from cases with the correct view
        tubedata = data.get_tube(tube)

        # get the referece if using reference initialization or sample based
        if selected_markers is None:
            used_channels = tubedata.markers
        else:
            used_channels = selected_markers[tube]
        treference = reference[tube]

        model = tfsom.SOMNodes(
            reference=treference,
            channels=used_channels,
            tube=tube,
            randnums=config("randnums"),
            preprocessing=config("dataset", "preprocessing"),
            **config("somnodes"),
            **config("tfsom"),
            tensorboard_dir=tensorboard_dir,
        )

        circ_buffer = collections.deque(maxlen=20)
        time_a = time.time()
        for label, randnum, result in model.transform_generator(tubedata):
            time_b = time.time()
            print(f"Saving {label} {randnum}")
            meta["label"].append(label)
            meta["randnum"].append(randnum)

            filename = f"{result.name}_t{tube}.csv"
            filepath = path / filename
            utils.save_csv(result, filepath)

            time_d = time_b - time_a
            circ_buffer.append(time_d)
            print(f"Training time: {time_d}s Rolling avg: {np.mean(circ_buffer)}s")
            time_a = time_b

    return pd.DataFrame(meta)
