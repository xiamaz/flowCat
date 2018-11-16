"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import os
import time
import contextlib
import collections
import argparse

import logging

import numpy as np
import pandas as pd

from sklearn import preprocessing

from flowcat.models import tfsom
from flowcat.data import case as ccase
from flowcat.data.case_dataset import CaseCollection, TubeView
from flowcat.configuration import Configuration, compare_configurations
from flowcat import utils


# choose another directory to save downloaded data
if "flowCat_tmp" in os.environ:
    utils.TMP_PATH = os.environ["flowCat_tmp"]


GROUPS = [
    "CLL", "PL", "FL", "HCL", "LPL", "MBL", "MCL", "MZL", "normal"
]


@contextlib.contextmanager
def timer(title):
    """Take the time for the enclosed block."""
    time_a = time.time()
    yield
    time_b = time.time()

    time_diff = time_b - time_a
    print(f"{title}: {time_diff:.3}s")


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


def generate_reference(args, all_config):
    """Generate a reference SOMmap using the given configuration."""
    config = all_config["reference"]

    reference_path = utils.URLPath(config["path"])
    reference_config = reference_path / "config.toml"

    # load existing if it already exists
    if reference_config.exists() and not args.recreate:
        old_config = Configuration.from_toml(reference_config)
        assert compare_configurations(config, old_config["reference"], section=None, method="left")
        reference_data = {
            t: utils.load_csv(reference_path / f"t{t}.csv") for t in config["view"]["tubes"]
        }
        return reference_data

    cases = CaseCollection.from_dir(config["cases"])
    labels = load_labels(config["labels"])
    data = cases.filter(labels=labels, **config["view"])

    config["selected_markers"] = data.selected_markers

    output_dir = utils.URLPath(config["path"])
    reference_data = {}
    for tube in data.selected_tubes:
        tubedata = data.get_tube(tube)
        marker_list = tubedata.markers

        model = tfsom.TFSom(channels=marker_list, tube=tube, **config["tfsom"], tensorboard_dir=args.tensorboard)

        # create a data generator
        datagen, length = tfsom.create_z_score_generator(tubedata.data, randnums=None)

        # train the network
        with timer("Training time"):
            model.train(datagen(), num_inputs=length)

        reference_map = pd.DataFrame(model.output_weights, columns=marker_list)

        # get the results and save them to files
        utils.save_csv(reference_map, output_dir / f"t{tube}.csv")
        reference_data[tube] = reference_map

    # Save the configuration if regenerated
    all_config.to_toml(reference_config)

    # return reference maps as dict of tubes
    return reference_data


def filter_tubedata_existing(tubedata, outdir):
    """Filter tubedata to remove all cases that have already been generated."""
    not_existing = []
    for tcase in tubedata:
        filename = f"{tcase.parent.id}_0_t{tubedata.tube}.csv"
        filepath = outdir / filename
        if not filepath.exists():
            not_existing.append(tcase)

    ne_tcases = TubeView(not_existing, markers=tubedata.markers, tube=tubedata.tube)
    return ne_tcases


def generate_soms(args, all_config, references, recreate=True):
    """Generate sommaps using the given configuration.
    Args:
        all_coufig: Configuration object
        references: Reference SOM dict.
        recreate: If true, existing files will be overwritten.
    """
    config = all_config["soms"]

    output_dir = utils.URLPath(config["path"])

    all_config.to_toml(output_dir + "_config.toml")

    cases = CaseCollection.from_dir(config["cases"])
    labels = load_labels(config["labels"])

    selected_markers = None
    if config["somnodes"]["initialization_method"] != "random":
        selected_markers = {k: list(v.columns) for k, v in references.items()}

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

        if not recreate:
            tubedata = filter_tubedata_existing(tubedata, output_dir)

        # get the referece if using reference initialization or sample based
        if config["somnodes"]["initialization_method"] == "random":
            reference = None
            used_channels = tubedata.markers
        else:
            print(references.keys())
            reference = references[tube]
            used_channels = list(reference.columns)

        model = tfsom.SOMNodes(
            reference=reference,
            channels=used_channels,
            **config["somnodes"],
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


def create_config():
    # START Configuration
    # General SOMmap
    c_general_gridsize = 32
    c_general_map_type = "toroid"
    # Data Configuration options
    c_general_cases = "s3://mll-flowdata/newCLL-9F"
    c_general_tubes = [1, 2]

    # Reference SOMmap options
    c_reference_name = "newselected1"
    c_reference_path = f"output/mll-sommaps/reference_maps/{c_reference_name}"
    c_reference_cases = c_general_cases
    c_reference_labels = "data/selected_cases.txt"
    c_reference_view = {
        "tubes": c_general_tubes,
        "num": 1,
        "groups": None,
        "infiltration": None,
        "counts": 8192,
    }
    c_reference_tfsom = {
        "model_name": c_reference_name,
        "m": c_general_gridsize, "n": c_general_gridsize,
        "map_type": c_general_map_type,
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

    # Individual SOMmap configuration
    c_soms_name = f"randnums_s{c_general_gridsize}_t{c_general_map_type}"
    c_soms_cases = c_general_cases
    c_soms_path = f"output/mll-sommaps/sample_maps/{c_soms_name}"
    c_soms_labels = None
    c_soms_view = {
        "tubes": c_general_tubes,
        "num": None,
        "groups": None,
        "infiltration": None,
        "counts": 8192,
    }
    c_soms_somnodes = {
        "m": c_general_gridsize, "n": c_general_gridsize,
        "map_type": c_general_map_type,
        "max_epochs": 10,
        "initialization_method": "reference",
        "counts": True,
        "subsample_size": 8192,
        "radius_cooling": "exponential",
        "learning_cooling": "exponential",
        "node_distance": "euclidean",
        "randnums": {
            "HCL": 11,
            "FL": 10,
            "MCL": 5,
            "PL": 4,
            "LPL": 4,
            "MZL": 3,
            "MBL": 2,
        },
        "fitmap_args": {
            "max_epochs": 2,
            "initial_learn": 0.1,
            "end_learn": 0.05,
            "initial_radius": 6,
            "end_radius": 1,
        },
        "model_name": c_soms_name,
    }

    # Output Configurations
    # END Configuration

    config = Configuration.from_localsdict(locals())
    return config


def load_config(path):
    if str(path).endswith(".json"):
        return Configuration.from_json(path)
    elif str(path).endswith(".toml"):
        return Configuration.from_toml(path)
    raise TypeError(f"Config file neither json nor toml format: {path}")


def save_config(args, config):
    if str(args.path).endswith(".json"):
        config.to_json(args.path)
    elif str(args.path).endswith(".toml"):
        config.to_toml(args.path)
    else:
        raise TypeError(f"Output fmt neither json nor toml format: {args.path}")


def create_som(args, config):
    reference_dict = generate_reference(args, config)
    generate_soms(args, config, reference_dict, recreate=args.recreate)


def main():
    configure_print_logging()

    parser = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    parser.add_argument("--config", help="Configuration file to load from.", type=utils.URLPath)
    parser.add_argument("--recreate", help="Recreate existing maps.", action="store_true")
    parser.add_argument("--tensorboard", help="Tensorboard directory", type=utils.URLPath)
    subparsers = parser.add_subparsers()

    parser_conf = subparsers.add_parser("config", help="Generate the config to a specified directory")
    parser_conf.add_argument("path", help="Output path to save configuration", type=utils.URLPath)
    parser_conf.set_defaults(fun=save_config)

    parser_create = subparsers.add_parser("som", help="Generate individual SOM, will also create reference if missing")
    parser_create.set_defaults(fun=create_som)

    parser_ref = subparsers.add_parser("reference", help="Generate reference SOM")
    parser_ref.set_defaults(fun=generate_reference)

    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = create_config()
    if hasattr(args, "fun"):
        args.fun(args, config)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
