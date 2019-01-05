"""
Functions to create SOMs.

These functions use configuration objects, as defined properly in configurations.
"""
import logging
import collections
import time

import numpy as np
import pandas as pd

from . import configuration, utils
from .dataset import case_dataset
from .models import tfsom


LOGGER = logging.getLogger(__name__)


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
                }
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
                }
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

    dataset = case_dataset.CaseView(cases, selected_markers=markers, selected_tubes=tubes)

    for tube in tubes:
        tubeview = dataset.get_tube(tube)
        yield tube, tfsom.create_z_score_generator(tubeview)


def create_som(cases, config, tensorboard_path=None, seed=None):
    """Create a SOM for the given list of cases with the configuration."""

    markers = config("dataset", "selected_markers")

    somweights = {}
    for tube, (datagen, length) in create_datasets(cases, config("dataset")):
        tmarkers = markers[tube]
        model = tfsom.TFSom(
            channels=tmarkers, tube=tube,
            **config("tfsom"),
            tensorboard_dir=tensorboard_path,
            seed=seed)
        # train the network
        with utils.timer("Training time"):
            model.train(datagen(), num_inputs=length)

        somweights[tube] = pd.DataFrame(model.output_weights, columns=tmarkers)

    return somweights


def load_som(path, tubes, suffix=False):
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
    for tube in tubes:
        if suffix:
            tpath = path + f"_t{tube}.csv"
        else:
            tpath = path / f"t{tube}.csv"
        data[tube] = utils.load_csv(tpath)
    return data


def save_som(data, path, suffix=False):
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
        if suffix:
            outpath = path + f"_t{tube}.csv"
        else:
            outpath = path / f"t{tube}.csv"
        LOGGER.debug("Saving SOM tube %d to %s", tube, outpath)
        utils.save_csv(somdata, outpath)


def load_reference(args):
    """Load a reference if it already exists in the given folder."""
    path = utils.URLPath(args.pathconfig("output", "som-reference"), args.refconfig("name"))

    if path.exists():
        print(f"Loading existing references in {path}")
        return load_som(path, args.refconfig("dataset", "filters", "tubes"), suffix=False)

    return None


def create_new_reference(args):
    """Create a new reference SOM."""
    config = args.refconfig
    print(f"Creating reference SOM with name {config('name')}")

    casespath = utils.get_path(config("dataset", "names", "FCS"), args.pathconfig("input", "FCS"))
    cases = case_dataset.CaseCollection.from_path(casespath)
    labels = utils.load_labels(config("dataset", "labels"))
    data = cases.filter(labels=labels, **config("dataset", "filters"))

    config.data["dataset"]["selected_markers"] = {str(k): v for k, v in data.selected_markers.items()}

    if args.tensorboard:
        tensorboard_path = args.tensorboard / args.name
        print(f"Creating tensorboard logs in {tensorboard_path}")
    else:
        tensorboard_path = None

    return create_som(data, config, tensorboard_path)


def save_reference(args, data):
    # Save reference data
    path = utils.URLPath(args.pathconfig("output", "som-reference"), args.refconfig("name"))
    print(f"Saving reference SOM in {path}")
    save_som(data, path, suffix=False)
    # Save reference configuration
    args.refconfig.to_file(get_config_path(path))


def generate_reference(args):
    """Generate a reference SOMmap using the given configuration."""
    # load existing if it already exists
    reference_data = load_reference(args)
    # create new reference
    if not reference_data:
        reference_data = create_new_reference(args)
        save_reference(args, reference_data)

    return reference_data


def filter_tubedata_existing(tubedata, outdir):
    """Filter tubedata to remove all cases that have already been generated."""
    not_existing = []
    for tcase in tubedata:
        filename = f"{tcase.parent.id}_0_t{tubedata.tube}.csv"
        filepath = outdir / filename
        if not filepath.exists():
            not_existing.append(tcase)

    ne_tcases = case_dataset.TubeView(not_existing, markers=tubedata.markers, tube=tubedata.tube)
    return ne_tcases


def generate_soms(args):
    """Generate sommaps using the given configuration."""
    config = args.somconfig
    output_dir = utils.URLPath(args.pathconfig("output", "som-sample"), config("name"))
    print(f"Create individual SOM in {output_dir}")
    config.to_file(output_dir / "config.toml")

    casespath = utils.get_path(config("dataset", "names", "FCS"), args.pathconfig("input", "FCS"))
    cases = case_dataset.CaseCollection.from_path(casespath)
    labels = utils.load_labels(config("dataset", "labels"))

    selected_markers = None
    if config("tfsom", "initialization_method") != "random":
        references = generate_reference(args)
        selected_markers = {k: list(v.columns) for k, v in references.items()}
        config.data["reference"] = args.refconfig.data["name"]
    else:
        references = {int(t): None for t in config("dataset", "filters", "tubes")}

    data = cases.filter(labels=labels, selected_markers=selected_markers, **config("dataset", "filters"))
    # get a dict how often specific groups should get replicated
    randnums = config("randnums")
    meta = {
        "label": [c.id for c in data for i in range(randnums.get(c.group, 1))],
        "randnum": [i for c in data for i in range(randnums.get(c.group, 1))],
        "group": [c.group for c in data for i in range(randnums.get(c.group, 1))],
    }

    for tube in data.selected_tubes:
        # get the correct data from cases with the correct view
        tubedata = data.get_tube(tube)

        if not args.no_recreate_samples:
            tubedata = filter_tubedata_existing(tubedata, output_dir)

        # get the referece if using reference initialization or sample based
        if selected_markers is None:
            used_channels = tubedata.markers
        else:
            used_channels = selected_markers[tube]
        reference = references[tube]

        model = tfsom.SOMNodes(
            reference=reference,
            channels=used_channels,
            tube=tube,
            **config("somnodes"),
            **config("tfsom"),
            tensorboard_dir=args.tensorboard,
        )

        circ_buffer = collections.deque(maxlen=20)
        time_a = time.time()
        for label, randnum, result in model.transform_generator(tubedata):
            time_b = time.time()
            print(f"Saving {label} {randnum}")

            filename = f"{result.name}_t{tube}.csv"
            filepath = output_dir / filename
            utils.save_csv(result, filepath)

            time_d = time_b - time_a
            circ_buffer.append(time_d)
            print(f"Training time: {time_d}s Rolling avg: {np.mean(circ_buffer)}s")
            time_a = time_b

    # save the metadata
    metadata = pd.DataFrame(meta)
    metadata.to_csv(f"{output_dir}.csv")
    return metadata
