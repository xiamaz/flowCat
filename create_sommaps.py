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
from flowcat.data.case_dataset import CaseCollection
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


def configure_print_logging(rootname="clustering"):
    """Configure default logging for visual output to stdout."""
    rootlogger = logging.getLogger(rootname)
    rootlogger.setLevel(logging.INFO)
    formatter = logging.Formatter()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)


def create_generator(data, transforms=None, fit_transformer=False):
    """Create a generator for the given data. Optionally applying additional transformations.
    Args:
        transforms: Optional transformation pipeline for the data.
        fit_transformer: Fit transformation pipeline to each new sample.
    Returns:
        Tuple of generator function and length of the data.
    """

    def generator_fun():
        for case in data:
            fcsdata = case.data
            # fcsdata = fcsdata.drop_columns("nix-APCA700")
            if transforms is not None:
                fcsdata = transforms.fit_transform(fcsdata) if fit_transformer else transforms.transform(fcsdata)
            yield fcsdata.data

    return generator_fun, len(data)


def create_z_score_generator(tubecases):
    """Normalize channel information for mean and standard deviation.
    Args:
        tubecases: List of tubecases.
    Returns:
        Generator function and length of tubecases.
    """

    transforms = ccase.FCSPipeline(steps=[
        ("zscores", ccase.FCSStandardScaler()),
        ("scale", ccase.FCSMinMaxScaler()),
    ])

    return create_generator(tubecases, transforms, fit_transformer=True)


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
        with open(str(path), "r") as f:
            labels = [l.strip() for l in f]
    return labels


def generate_reference(all_config):
    """Generate a reference SOMmap using the given configuration."""
    config = all_config["reference"]

    reference_path = utils.URLPath(config["path"])
    reference_config = reference_path / "config.toml"

    # load existing if it already exists
    if reference_config.exists():
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
    reference_maps = {}
    for tube in data.selected_tubes:
        tubedata = data.get_tube(tube)
        marker_list = tubedata.markers

        model = tfsom.TFSom(channels=marker_list, **config["tfsom"])

        # create a data generator
        datagen, length = create_z_score_generator(tubedata.data)

        # train the network
        with timer("Training time"):
            model.train(datagen(), num_inputs=length)

        reference_map = pd.DataFrame(model.output_weights, columns=marker_list)

        # get the results and save them to files
        utils.save_csv(reference_map, output_dir / f"t{tube}.csv")
        reference_maps[tube] = reference_map

    # Save the configuration if regenerated
    all_config.to_toml(reference_config)

    # return reference maps as dict of tubes
    return reference_map


def generate_soms(all_config, references):
    """Generate sommaps using the given configuration."""
    config = all_config["soms"]

    output_dir = utils.URLPath(config["path"])

    all_config.to_toml(output_dir / "config.toml")

    cases = CaseCollection.from_dir(config["cases"])
    labels = load_labels(config["labels"])
    data = cases.filter(labels=labels, **config["view"])
    meta = {
        "label": [c.id for c in data.data],
        "group": [c.group for c in data.data],
    }

    for tube in data.selected_tubes:
        # get the correct data from cases with the correct view
        tubedata = data.get_tube(tube)

        # get the referece if using reference initialization or sample based
        if config["somnodes"]["initialization_method"] == "random":
            reference = None
            used_channels = tubedata.markers
        else:
            reference = references[tube]
            used_channels = list(reference.columns)

        model = tfsom.SOMNodes(
            reference=reference,
            channels=used_channels,
            **config["somnodes"],
        )

        datagen, _ = create_z_score_generator(tubedata.data)

        tubelabels = [c.parent.id for c in tubedata]
        circ_buffer = collections.deque(maxlen=20)
        time_a = time.time()
        for label, result in zip(tubelabels, model.transform(datagen())):
            time_b = time.time()
            print(f"Saving {label}")
            filename = f"{label}_t{tube}.csv"
            filepath = output_dir / filename
            utils.save_csv(result,filepath)
            time_d = time_b - time_a
            circ_buffer.append(time_d)
            print(f"Training time: {time_d}s Rolling avg: {np.mean(circ_buffer)}s")
            time_a = time_b

    # save the metadata
    metadata = pd.DataFrame(meta)
    metadata.to_csv(f"{output_dir}.csv")
    return metadata


def main():
    # START Configuration
    # General SOMmap
    c_general_gridsize = 32
    c_general_map_type = "toroid"
    # Data Configuration options
    c_general_cases = "s3://mll-flowdata/CLL-9F"
    c_general_tubes = [1, 2]

    # Reference SOMmap options
    c_reference_name = "testreference"
    c_reference_path = f"mll-sommaps/reference_maps/{c_reference_name}"
    c_reference_cases = c_general_cases
    c_reference_labels = "data/selected_cases.txt"
    c_reference_view = {
        "tubes": c_general_tubes,
        "num": 1,
        "groups": None,
        "infiltration": None,
        "counts": None,
    }
    c_reference_tfsom = {
        "model_name": c_reference_name,
        "m": c_general_gridsize, "n": c_general_gridsize,
        "map_type": c_general_map_type,
        "max_epochs": 10,
        "batch_size": 1,
        "subsample_size": 2048,
        "initial_learning_rate": 0.5,
        "end_learning_rate": 0.1,
        "learning_cooling": "exponential",
        "initial_radius": int(c_general_gridsize / 2),
        "end_radius": 1,
        "radius_cooling": "exponential",
        "node_distance": "euclidean",
        "initialization_method": "random",
        "tensorboard_dir": None,
    }

    # Individual SOMmap configuration
    c_soms_name = f"testrun_s{c_general_gridsize}_t{c_general_map_type}"
    c_soms_cases = c_general_cases
    c_soms_path = f"mll-sommaps/sample_maps/{c_soms_name}"
    c_soms_labels = "data/selected_cases.txt"
    c_soms_view = {
        "tubes": c_general_tubes,
        "num": 50,
        "groups": None,
        "infiltration": None,
        "counts": None,
    }
    c_soms_somnodes = {
        "m": c_general_gridsize, "n": c_general_gridsize,
        "map_type": c_general_map_type,
        "max_epochs": 10,
        "initialization_method": "reference",
        "counts": True,
        "subsample_size": 2048,
        "radius_cooling": "exponential",
        "learning_cooling": "exponential",
        "node_distance": "euclidean",
        "fitmap_args": {
            "max_epochs": 2,
            "initial_learn": 0.1,
            "end_learn": 0.05,
            "initial_radius": 6,
            "end_radius": 1,
        },
        "model_name": c_soms_name,
        "tensorboard_dir": None,
    }

    # Output Configurations
    # END Configuration

    config = Configuration.from_localsdict(locals())

    configure_print_logging()

    parser = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    parser.add_argument("--references", help="Generate references only.", action="store_true")
    args = parser.parse_args()

    reference_dict = generate_reference(config)
    if args.references:
        print("Only generating references. Not generating individual SOMs.")
    else:
        generate_soms(config, reference_dict)


if __name__ == "__main__":
    main()
