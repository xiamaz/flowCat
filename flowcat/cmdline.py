import json
import logging
from collections import defaultdict

from flowcat import utils, io_functions, sommodels
from flowcat.dataset import case_dataset
from argmagic import argmagic


LOGGER = logging.getLogger(__name__)


def setup_logging():
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger("flowcat", handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def filter(
        data: utils.URLPath,
        meta: utils.URLPath,
        output: utils.URLPath,
        filters: json.loads,
        sample: int = 0):
    """Filter data on the given filters and output resulting dataset metadata
    to destination.

    Args:
        data: Path to fcs data.
        meta: Path to dataset metadata.
        output: Path to output for metadata.
        filters: Filters for individual cases.
        sample: Number of cases per group.
    """
    try:
        dataset = io_functions.load_case_collection(data, meta)
    except TypeError:
        dataset = io_functions.load_case_collection_from_caseinfo(data, meta)
    dataset = dataset.filter(**filters)
    if sample:
        dataset = dataset.sample(sample)
    print("Saving", dataset)
    io_functions.save_case_collection(dataset, output)


def reference(
        data: utils.URLPath,
        meta: utils.URLPath,
        output: utils.URLPath,
        labels: utils.URLPath,
        tensorboard: bool = False,
        trainargs: json.loads = None,
        selected_markers: json.loads = None):
    """Train new reference SOM from random using data filtered by labels."""
    setup_logging()

    dataset = io_functions.load_case_collection(data, meta)
    labels = io_functions.load_json(labels)
    dataset = dataset.filter(labels=labels)

    if trainargs is None:
        trainargs = {
            "marker_name_only": False,
            "max_epochs": 10,
            "batch_size": 50000,
            "initial_radius": 16,
            "end_radius": 2,
            "radius_cooling": "linear",
            # "marker_images": sommodels.fcssom.MARKER_IMAGES_NAME_ONLY,
            "map_type": "toroid",
            "dims": (32, 32, -1),
            "scaler": "MinMaxScaler",
        }

    if selected_markers is None:
        selected_markers = dataset.selected_markers

    tensorboard_dir = None
    if tensorboard:
        tensorboard_dir = output / "tensorboard"

    print("Creating SOM model with following parameters:")
    print(trainargs)
    print(selected_markers)
    model = sommodels.casesom.CaseSom(
        tubes=selected_markers,
        tensorboard_dir=tensorboard_dir,
        modelargs=trainargs,
    )
    print(f"Training SOM")
    model.train(dataset)

    print(f"Saving trained SOM to {output}")
    io_functions.save_casesom(model, output)


def transform(
        data: utils.URLPath,
        meta: utils.URLPath,
        output: utils.URLPath,
        reference: utils.URLPath,
        transargs: json.loads = None,
        sample: int = 0):
    """Transform dataset using a reference SOM."""
    dataset = io_functions.load_case_collection(data, meta)

    # randomly sample 'sample' number cases from each group
    if sample:
        dataset = dataset.sample(sample)

    if transargs is None:
        transargs = {
            "max_epochs": 4,
            "batch_size": 50000,
            "initial_radius": 4,
            "end_radius": 1,
        }

    print(f"Loading referece from {reference}")
    model = io_functions.load_casesom(reference, **transargs)

    print(f"Trainsforming individual samples")
    output.mkdir()
    casesamples = defaultdict(list)
    count_samples = len(dataset) * len(model.models)
    countlen = len(str(count_samples))
    for i, (case, somsample) in enumerate(utils.time_generator_logger(model.transform_generator(dataset))):
        sompath = output / f"{case.id}_t{somsample.tube}.npy"
        io_functions.save_som(somsample.data, sompath, save_config=False)
        somsample.data = None
        somsample.path = sompath
        casesamples[case.id].append(somsample)
        print(f"[{str(i + 1).rjust(countlen, ' ')}/{count_samples}] Created tube {somsample.tube} for {case.id}")

    print(f"Saving result to new collection at {output}")
    som_dataset = case_dataset.CaseCollection([
        case.copy(samples=casesamples[case.id])
        for case in dataset
    ])
    som_dataset.selected_markers = {
        m.tube: m.model.markers for m in model.models.values()
    }
    io_functions.save_case_collection(som_dataset, output + ".json.gz")
    io_functions.save_json(
        {
            tube: {
                "dims": m.model.dims,
                "channels": m.model.markers,
            } for tube, m in model.models.items()
        }, output + "_config.json")


def train(data: utils.URLPath, output: utils.URLPath):
    """Train a new classifier using SOM data."""
    raise NotImplementedError


def predict(case, model: utils.URLPath, output: utils.URLPath):
    """Generate predictions and plots for a single case.

    Args:
        case: Single case with FCS files.
        model: Path to model containing CNN and SOMs.
        output: Destination for plotting.
    """
    raise NotImplementedError


def main():
    argmagic([predict, train, filter, reference, transform])
