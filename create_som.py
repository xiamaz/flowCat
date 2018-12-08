#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import time
import collections
import argparse

import logging

import numpy as np
import pandas as pd

from flowcat import utils, mappings, configuration
from flowcat.models import tfsom
from flowcat.dataset import case_dataset


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    rootlogger = logging.getLogger(rootname)
    rootlogger.setLevel(logging.INFO)
    formatter = logging.Formatter()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)


def load_labels(path):
    """Load list of labels. Either in .json, .p (pickle) or .txt format.
    Args:
        path: path to label file.
    """
    if not path:
        return path

    path = utils.URLPath(path)
    try:
        labels = utils.load_file(path)
    except TypeError:
        # try loading as simple txt file instead
        with open(str(path), "r") as f:
            labels = [l.strip() for l in f]
    return labels


def get_filtered_data(config):
    """Create a filtered case view from the given configuration."""
    return data


def get_config_path(path):
    """Get configuration for the given dataset."""
    return path / "config.toml"


def get_config(path, section=None):
    """Load configuration in the given dataset and return the given section."""
    config = Configuration.from_file(get_config_path(path))
    return config if section is None else config[section]


def load_reference_if_same(path, new_config):
    """Check whether the given path contains a dataset generated with the same
    configuration. Return the reference dict if yes, otherwise return None."""

    # skip if dataset folder does not exist
    if not path.exists():
        return None

    # check if configuration is the same
    ref_config = get_config(path)
    if not configuration.compare_configurations(
            new_config["reference"],
            ref_config["reference"],
            section=None, method="left"
    ):
        if not utils.CLOBBER:
            raise RuntimeError(f"{path} exists and will not be overwritten because CLOBBER is {utils.CLOBBER}")
        return None

    return {
        t: utils.load_csv(path / f"t{t}.csv") for t in config["reference"]["view"]["tubes"]
    }


def create_new_reference(path, config):
    output_dir = utils.URLPath(config["path"])
    reference_data = {}

    cases = case_dataset.CaseCollection.from_path(config["cases"])
    labels = load_labels(config["labels"])
    data = cases.filter(labels=labels, **config["view"])

    config["selected_markers"] = data.selected_markers

    for tube in data.selected_tubes:
        tubedata = data.get_tube(tube)
        marker_list = tubedata.markers

        model = tfsom.TFSom(channels=marker_list, tube=tube, **config["tfsom"], tensorboard_dir=args.tensorboard)

        # create a data generator
        datagen, length = tfsom.create_z_score_generator(tubedata.data, randnums=None)

        # train the network
        with utils.timer("Training time"):
            model.train(datagen(), num_inputs=length)

        reference_map = pd.DataFrame(model.output_weights, columns=marker_list)

        # get the results and save them to files
        reference_data[tube] = reference_map

    return reference_data


def save_reference(config, data, path):
    # Save reference data
    for tube, reference_map in data.items():
        utils.save_csv(reference_map, path / f"t{tube}.csv")

    # Save reference configuration
    ref_only = all_config.copy()
    del ref_only["som"]
    ref_only.to_toml(get_config_path(path))


def generate_reference(args):
    """Generate a reference SOMmap using the given configuration."""
    all_config = args.refconfig
    config = all_config["reference"]
    reference_path = utils.URLPath(config["path"])

    # load existing if it already exists
    reference_data = load_reference_if_same(reference_path, all_config)
    # create new reference
    if not reference_data:
        reference_data = create_new_reference(reference_path, config)
        save_reference(all_config, reference_data, reference_path)

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
    """Generate sommaps using the given configuration.
    Args:
        all_coufig: Configuration object
        references: Reference SOM dict.
    """
    all_config = args.somconfig
    config = all_config["soms"]

    output_dir = utils.URLPath(config["path"])

    all_config.to_toml(output_dir / "config.toml")

    cases = case_dataset.CaseCollection.from_path(config["cases"])
    labels = load_labels(config["labels"])

    selected_markers = None
    if config["somnodes"]["initialization_method"] != "random":
        references = generate_reference(args, all_config)
        selected_markers = {k: list(v.columns) for k, v in references.items()}
    else:
        references = {int(t): None for t in config["view"]["tubes"]}

    data = cases.filter(labels=labels, selected_markers=selected_markers, **config["view"])
    if "randnums" in config["somnodes"]:
        randnums = config["somnodes"]["randnums"]
    else:
        randnums = {}
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
            **config["somnodes"],
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


def create_reference_config():
    """Create a reference configuration."""
    # Reference SOMmap options
    c_general_name = "testrun"
    c_dataset_labels = "data/selected_cases.txt"
    c_dataset_names = {
        "FCS": "fixedCLL-9F",
    }
    c_dataset_filters = {
        "tubes": [1, 2],
        "num": 1,
        "groups": None,
        "infiltration": None,
        "counts": 8192,
    }
    c_model_tfsom = {
        "model_name": c_general_name,
        "m": 32, "n": 32,
        "map_type": "toroid",
        "max_epochs": 10,
        "batch_size": 1,
        "subsample_size": 8192,
        "initial_learning_rate": 0.5,
        "end_learning_rate": 0.1,
        "learning_cooling": "exponential",
        "initial_radius": int(c_general_gridsize / 2),
        "end_radius": 1,
        "radius_cooling": "exponential",
        "node_distance": "euclidean",
        "initialization_method": "random",
    }
    config = configuration.Configuration.from_localsdict(locals(), section="c")
    return config


def create_som_config():
    # START Configuration
    # General SOMmap
    c_general_name = "testrun"
    # Data Configuration options
    c_dataset_labels = None
    c_dataset_names = {
        "FCS": "fixedCLL-9F",
    }
    c_dataset_filters = {
        "tubes": [1, 2],
        "num": None,
        "groups": None,
        "infiltration": None,
        "counts": 8192,
    }
    # Individual SOMmap configuration
    c_soms_somnodes = {
        "m": 32, "n": 32,
        "map_type": "toroid",
        "max_epochs": 10,
        "initialization_method": "random",
        "counts": True,
        "subsample_size": 8192,
        "radius_cooling": "exponential",
        "learning_cooling": "exponential",
        "node_distance": "euclidean",
        "randnums": {},
        "fitmap_args": {
            "max_epochs": 10,
            "initial_learn": 0.5,
            "end_learn": 0.1,
            "initial_radius": 16,
            "end_radius": 1,
        },
        "model_name": c_general_name,
    }
    # Output Configurations
    # END Configuration

    # Only use c_ prefixed variables for configuration generation
    config = configuration.Configuration.from_localsdict(locals(), section="c")
    return config


def main():
    configure_print_logging()

    parser = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    parser.add_argument(
        "--somconfig",
        help="Individual SOM config file.",
        type=configuration.configuration_builder(create_som_config),
        default="")
    parser.add_argument(
        "--refconfig",
        help="Reference SOM config file.",
        type=configuration.configuration_builder(create_reference_config),
        default="")
    parser.add_argument(
        "--tensorboard",
        help="Tensorboard directory",
        type=utils.URLPath)
    parser.add_argument(
        "--name",
        help="Name of the current run. Will be used as output folder name.",
        type=str)
    subparsers = parser.add_subparsers()

    parser_conf = subparsers.add_parser("config", help="Generate the config to a specified directory")
    parser_conf.add_argument(
        "--type",
        choices=["refconfig", "somconfig"],
        help="Generate reference or individual SOM config")
    parser_conf.add_argument(
        "path",
        help="Output path to save configuration",
        type=utils.URLPath)
    parser_conf.set_defaults(
        fun=lambda args: getattr(args, args.type).to_file(args.path))

    parser_create = subparsers.add_parser("som", help="Generate individual SOM, will also create reference if missing")
    parser_create.add_argument(
        "--no-recreate-samples", help="Do not regenerate already created individual samples", action="store_true")
    parser_create.set_defaults(fun=generate_soms)

    parser_ref = subparsers.add_parser("reference", help="Generate reference SOM")
    parser_ref.set_defaults(fun=generate_reference)

    args = parser.parse_args()
    if hasattr(args, "fun"):
        args.fun(args)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
    # create_group_maps("tensorboard_test")
