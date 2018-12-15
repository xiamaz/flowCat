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

from flowcat import utils, configuration
from flowcat.models import tfsom
from flowcat.dataset import case_dataset


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


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


def load_reference_if_same(args):
    """Check whether the given path contains a dataset generated with the same
    configuration. Return the reference dict if yes, otherwise return None."""

    path = utils.URLPath(args.pathconfig("output", "som-reference"), args.refconfig("name"))

    # skip if dataset folder does not exist
    if not path.exists():
        return None

    # check if configuration is the same
    old_config = ReferenceConfig.from_file(get_config_path(path))
    if args.refconfig == old_config:
        print(f"Loading existing references in {path}")
        return {
            t: utils.load_csv(path / f"t{t}.csv") for t in config("dataset", "filters", "tubes")
        }
    else:
        if not utils.CLOBBER:
            raise RuntimeError(f"{path} exists with different configuration and will not be overwritten because CLOBBER is {utils.CLOBBER}")
        print(f"Old data is different. Clobbering {path}")
        return None


def create_new_reference(args):
    config = args.refconfig
    reference_data = {}
    print(f"Creating reference SOM with name {config('name')}")

    casespath = utils.get_path(config("dataset", "names", "FCS"), args.pathconfig("input", "FCS"))
    cases = case_dataset.CaseCollection.from_path(casespath)
    labels = load_labels(config("dataset", "labels"))
    data = cases.filter(labels=labels, **config("dataset", "filters"))

    config.data["dataset"]["selected_markers"] = {str(k): v for k, v in data.selected_markers.items()}

    if args.tensorboard:
        tensorboard_path = args.tensorboard / args.name
        print(f"Creating tensorboard logs in {tensorboard_path}")
    else:
        tensorboard_path = None

    for tube in data.selected_tubes:
        tubedata = data.get_tube(tube)
        marker_list = tubedata.markers

        model = tfsom.TFSom(channels=marker_list, tube=tube, **config("tfsom"), tensorboard_dir=tensorboard_path)

        # create a data generator
        datagen, length = tfsom.create_z_score_generator(tubedata.data, randnums=None)

        # train the network
        with utils.timer("Training time"):
            model.train(datagen(), num_inputs=length)

        reference_map = pd.DataFrame(model.output_weights, columns=marker_list)

        # get the results and save them to files
        reference_data[tube] = reference_map

    return reference_data


def save_reference(args, data):
    # Save reference data
    path = utils.URLPath(args.pathconfig("output", "som-reference"), args.refconfig("name"))
    print(f"Saving reference SOM in {path}")
    for tube, reference_map in data.items():
        utils.save_csv(reference_map, path / f"t{tube}.csv")
    # Save reference configuration
    args.refconfig.to_file(get_config_path(path))


def generate_reference(args):
    """Generate a reference SOMmap using the given configuration."""
    # load existing if it already exists
    reference_data = load_reference_if_same(args)
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
    labels = load_labels(config("dataset", "labels"))

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


class ReferenceConfig(configuration.SOMConfig):

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


def run_config(args):
    config = getattr(args, args.type)
    if args.output:
        config.to_file(utils.URLPath(args.output))
    else:
        print(config)


def main():
    configure_print_logging()

    parser = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    parser.add_argument("--somconfig", help="Individual SOM config file.", default="")
    parser.add_argument("--refconfig", help="Reference SOM config file.", default="")
    parser.add_argument(
        "--pathconfig",
        help="Paths configuration.",
        type=lambda p: configuration.PathConfig.from_file(p) if p else configuration.PathConfig({}),
        default="paths.toml")
    parser.add_argument(
        "--tensorboard",
        help="Tensorboard directory",
        type=utils.URLPath)
    parser.add_argument(
        "--name",
        help="Name of the current run. Will be used as output folder name.",
        type=str, default="testname")
    subparsers = parser.add_subparsers()

    parser_conf = subparsers.add_parser("config", help="Generate the config to a specified directory")
    parser_conf.add_argument(
        "--type",
        choices=["refconfig", "somconfig", "pathconfig"],
        help="Generate reference or individual SOM config",
        default="refconfig")
    parser_conf.add_argument(
        "-o", "--output",
        help="Output path to save configuration",
        default="")
    parser_conf.set_defaults(fun=run_config)

    parser_create = subparsers.add_parser("som", help="Generate individual SOM, will also create reference if missing")
    parser_create.add_argument(
        "--no-recreate-samples", help="Do not regenerate already created individual samples", action="store_true")
    parser_create.set_defaults(fun=generate_soms)

    parser_ref = subparsers.add_parser("reference", help="Generate reference SOM")
    parser_ref.set_defaults(fun=generate_reference)

    args = parser.parse_args()

    if args.refconfig:
        args.refconfig = ReferenceConfig.from_file(args.refconfig)
    else:
        args.refconfig = ReferenceConfig.generate_config(args)
    if args.somconfig:
        args.somconfig = IndivConfig.from_file(args.somconfig)
    else:
        args.somconfig = IndivConfig.generate_config(args)

    if hasattr(args, "fun"):
        args.fun(args)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
    # create_group_maps("tensorboard_test")
