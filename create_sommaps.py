"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import os
import time
import pathlib
import contextlib
import collections
import argparse

import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn import preprocessing

from flowcat.models import tfsom
from flowcat.data.collection import CaseCollection
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
            fcsdata.drop("nix-APCA700", axis=1, inplace=True, errors="ignore")
            if transforms is not None:
                trans = transforms.fit_transform(fcsdata) if fit_transformer else transforms.transform(fcsdata)
                fcsdata = pd.DataFrame(trans, columns=fcsdata.columns)
            yield fcsdata

    return generator_fun, len(data)


def create_z_score_generator(tubecases):
    """Normalize channel information for mean and standard deviation.
    Args:
        tubecases: List of tubecases.
    Returns:
        Generator function and length of tubecases.
    """

    transforms = Pipeline(steps=[
        ("zscores", preprocessing.StandardScaler()),
        ("scale", preprocessing.MinMaxScaler()),
    ])

    return create_generator(tubecases, transforms, fit_transformer=True)


def load_labels(path):
    """Load list of labels. Either in .json, .p (pickle) or .txt format.
    Args:
        path: path to label file.
    """
    if not path:
        return path

    path = pathlib.Path(path)
    if path.suffix == ".json":
        labels = utils.load_json(path)
    elif path.suffix == ".p":
        labels = utils.load_pickle(path)
    else:
        with open(str(path), "r") as f:
            labels = [l.strip() for l in f]
    return labels


def load_view(config, section, cases):
    caseview = cases.create_view(
        labels=load_labels(config[f"c_{section}_data_labels"]),
        num=config[f"c_{section}_data_num"],
        groups=config[f"c_{section}_data_groups"],
        infiltration=config[f"c_{section}_data_infiltration"],
        counts=config[f"c_{section}_data_counts"],
    )
    return caseview


def load_cases(config):
    """Load a case collection object using the given config object."""
    cases = CaseCollection(inputpath=config["c_cases_path"], tubes=config["c_tubes"])
    return cases


def generate_reference(config):
    """Generate a reference SOMmap using the given configuration."""
    cases = load_cases(config)
    data = load_view(config, "reference", cases)  # select the required subselection

    output_dir = pathlib.Path(config["c_reference_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_maps = {}
    for tube in config["c_tubes"]:
        tubedata = data.get_tube(tube)
        marker_list = tubedata.markers

        model = tfsom.TFSom(
            m=config["c_general_gridsize"],
            n=config["c_general_gridsize"],
            channels=marker_list,
            map_type=config["c_general_maptype"],
            subsample_size=config["c_reference_subsample_size"],
            batch_size=config["c_reference_batch_size"],
            initial_learning_rate=config["c_reference_initial_learning_rate"],
            end_learning_rate=config["c_reference_end_learning_rate"],
            initial_radius=config["c_reference_initial_radius"],
            end_radius=config["c_reference_end_radius"],
            radius_cooling=config["c_reference_radius_cooling"],
            learning_cooling=config["c_reference_learning_cooling"],
            node_distance=config["c_reference_node_distance"],
            max_epochs=config["c_reference_max_epochs"],
            initialization_method=config["c_reference_initialization_method"],
            model_name=config["c_reference_model_name"],
            tensorboard_dir=config["c_reference_tensorboard_dir"],
        )

        # create a data generator
        datagen, length = create_z_score_generator(tubedata.data)

        # train the network
        with timer("Training time"):
            model.train(datagen(), num_inputs=length)

        reference_map = pd.DataFrame(model.output_weights, columns=marker_list)

        # get the results and save them to files
        reference_map.to_csv(output_dir / f"t{tube}.csv")
        reference_maps[tube] = reference_map

    # return reference maps as dict of tubes
    return reference_map


def generate_sommaps(config, references):
    """Generate sommaps using the given configuration."""
    output_dir = pathlib.Path(config["c_sommaps_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    utils.save_json(config, output_dir / "config.json", clobber=False)

    cases = load_cases(config)
    data = load_view(config, "sommaps", cases)  # select the required subselection
    meta = {
        "label": [c.id for c in data.data],
        "group": [c.group for c in data.data],
    }

    for tube in config["c_tubes"]:
        # get the correct data from cases with the correct view
        tubedata = data.get_tube(tube)

        # get the referece if using reference initialization or sample based
        if config["c_sommaps_initialization_method"] == "random":
            reference = None
            used_channels = tubedata.markers
        else:
            reference = references[tube]
            used_channels = list(reference.columns)

        model = tfsom.SOMNodes(
            m=config["c_general_gridsize"], n=config["c_general_gridsize"], channels=used_channels,
            map_type=config["c_general_maptype"],
            initialization_method=config["c_sommaps_initialization_method"], reference=reference,
            counts=config["c_sommaps_counts"],
            subsample_size=config["c_sommaps_subsample_size"],
            radius_cooling=config["c_sommaps_radius_cooling"],
            learning_cooling=config["c_sommaps_learning_cooling"],
            node_distance=config["c_sommaps_node_distance"],
            fitmap_args=config["c_sommaps_fitmap_args"],
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
            result.to_csv(filepath)
            time_d = time_b - time_a
            circ_buffer.append(time_d)
            print(f"Training time: {time_d}s Rolling avg: {np.mean(circ_buffer)}s")
            time_a = time_b

    # save the metadata
    metadata = pd.DataFrame(meta)
    metadata.to_csv(f"{output_dir}.csv")
    return metadata


def check_configuration(path, keytypes, curdata):
    """Check that old and new configuration match in the specified keytypes."""
    # grab all keynames that are used for comparison
    selected_keys = [k for k in curdata if any(kk in k for kk in keytypes)]

    olddata = utils.load_json(path / "config.json")
    for key in selected_keys:
        if str(olddata[key]) != str(curdata[key]):
            print(f"{olddata[key]} != {curdata[key]}")
            return False
    return True


def load_reference(path, tubes):
    """Load references for the given tubes into a dict of dataframes."""
    return {
        t: pd.read_csv(path / f"t{t}.csv", index_col=0) for t in tubes
    }


def create_or_load_reference(config):
    """Create or load a reference file based on the given config."""
    reference_path = pathlib.Path(config["c_reference_path"])

    if (reference_path / "config.json").exists():
        if not check_configuration(
                reference_path, ["general", "reference"], config):
            raise RuntimeError(f"{reference_path} configuration has changed.")
        reference_data = load_reference(reference_path, config["c_tubes"])
    else:
        reference_data = generate_reference(config)
        # Save the configuration if regenerated
        utils.save_json(config, reference_path / "config.json", clobber=False)

    return reference_data


def main():
    ## START Configuration
    # General SOMmap
    c_general_gridsize = 32
    c_general_maptype = "toroid"

    # Reference SOMmap options
    c_reference_name = "testreference"
    c_reference_max_epochs = 10

    c_reference_data_labels = "data/selected_cases.txt"
    c_reference_data_num = 1  # per cohort number
    c_reference_data_groups = None  # cohorts to be included
    c_reference_data_infiltration = None  # minimum infiltration for usage
    c_reference_data_counts = None

    c_reference_batch_size = 1
    c_reference_subsample_size = 2048
    c_reference_initial_learning_rate = 0.5
    c_reference_end_learning_rate = 0.1
    c_reference_initial_radius = int(c_general_gridsize / 2)  # defaults to half of gridsize
    c_reference_end_radius = 1
    c_reference_radius_cooling = "exponential"
    c_reference_learning_cooling = "exponential"
    c_reference_node_distance = "euclidean"
    c_reference_max_epochs = 10
    c_reference_initialization_method = "random"
    c_reference_model_name = c_reference_name
    # c_reference_tensorboard_dir = f"tensorboard/ref_{c_reference_model_name}"
    c_reference_tensorboard_dir = None

    c_reference_path = f"mll-sommaps/reference_maps/{c_reference_name}"

    # Individual SOMmap configuration
    c_sommaps_name = f"testrun_s{c_general_gridsize}_t{c_general_maptype}"
    c_sommaps_max_epochs = 10

    c_sommaps_data_labels = None
    c_sommaps_data_num = 50  # per cohort number
    c_sommaps_data_groups = None  # cohorts to be included
    c_sommaps_data_infiltration = None  # minimum infiltration for usage
    c_sommaps_data_counts = None

    c_sommaps_batch_size = 1
    c_sommaps_counts = True
    c_sommaps_subsample_size = 2048
    c_sommaps_radius_cooling = "exponential"
    c_sommaps_learning_cooling = "exponential"
    c_sommaps_node_distance = "euclidean"
    c_sommaps_initialization_method = "reference"

    c_sommaps_fitmap_args = {
        "max_epochs": 2,
        "initial_learn": 0.1,
        "end_learn": 0.05,
        "initial_radius": 6,
        "end_radius": 1,
    }

    c_sommaps_model_name = c_sommaps_name
    c_sommaps_tensorboard = False
    c_sommaps_tensorboard_dir = f"tensorboard/ref_{c_sommaps_model_name}"
    c_sommaps_path = f"mll-sommaps/sample_maps/{c_sommaps_name}"

    # Data Configuration options
    c_cases_path = "s3://mll-flowdata/CLL-9F"
    c_tubes = [1, 2]

    # Output Configurations
    ## END Configuration

    config_dict = locals()
    configure_print_logging()

    parser = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    parser.add_argument("--references", help="Generate references only.", action="store_true")
    args = parser.parse_args()

    reference_dict = create_or_load_reference(config_dict)

    if args.references:
        print("Only generating references. Not generating individual SOMs.")
    else:
        generate_sommaps(config_dict, reference_dict)


if __name__ == "__main__":
    main()
