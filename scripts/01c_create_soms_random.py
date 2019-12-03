#!/usr/bin/env python3
"""
Create SOMs that are always reinitialized from random initial values.
"""
import argparse
import logging
from collections import defaultdict

import pandas as pd

from flowcat import utils, io_functions, parser
from flowcat.dataset import sample, case_dataset
from flowcat.sommodels import casesom


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def transform_cases(dataset, model, output):
    """Create individidual SOMs for all cases in the dataset.
    Args:
        dataset: CaseIterable with a number of cases, for which SOMs should be
                 generated.
        model: Model with initial weights, which should be used for generation
               of SOMs.
        output: Output directory for SOMs

    Returns:
        Nothing.
    """
    output.mkdir()
    casesamples = defaultdict(list)
    for case, somsample in utils.time_generator_logger(model.transform_generator(dataset)):
        sompath = output / f"{case.id}_t{somsample.tube}.npy"
        io_functions.save_som(somsample.data, sompath, save_config=False)
        somsample.data = None
        somsample.path = sompath
        casesamples[case.id].append(somsample)

    somcases = []
    for case in dataset:
        somcases.append(case.copy(samples=casesamples[case.id]))

    somcollection = case_dataset.CaseCollection(somcases)
    io_functions.save_json(somcollection, output + ".json")

    labels = [{"label": case.id, "randnum": 0, "group": case.group} for case in dataset]
    # Save metadata into an additional csv file with the same name
    metadata = pd.DataFrame(labels)
    io_functions.save_csv(metadata, output + ".csv")
    io_functions.save_json(
        {
            tube: {
                "dims": m.model.dims,
                "channels": m.model.markers,
            } for tube, m in model.models.items()
        }, output + "_config.json")


def main(args):
    """Load a model with given transforming arguments and transform individual
    cases."""
    cases = io_functions.load_case_collection(args.data, args.meta)
    # cases = cases.sample(1, groups=["CLL", "normal"])
    selected_markers = cases.selected_markers
    marker_name_only = False

    if args.tensorboard:
        tensorboard_dir = args.output / "tensorboard"
    else:
        tensorboard_dir = None

    # scaler = "RefitMinMaxScaler"
    scaler = args.scaler
    # Training parameters for the model can be respecified, the only difference
    # between transform and normal traninig, is that after a transformation is
    # completed, the original weights will be restored to the model.
    model = casesom.CaseSom(
        tubes=selected_markers,
        tensorboard_dir=tensorboard_dir,
        modelargs={
            "marker_name_only": marker_name_only,
            "max_epochs": 5,
            "batch_size": 50000,
            "initial_radius": int(args.size / 2),
            "end_radius": 1,
            "radius_cooling": "linear",
            # "marker_images": sommodels.fcssom.MARKER_IMAGES_NAME_ONLY,
            "map_type": "toroid",
            "dims": (args.size, args.size, -1),
            "scaler": scaler,
        }
    )

    transform_cases(cases, model, args.output)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    PARSER = parser.add_dataset_args(PARSER)
    PARSER.add_argument(
        "--tensorboard",
        help="Flag to enable tensorboard logging",
        action="store_true")
    PARSER.add_argument(
        "--size",
        default=32,
        type=int)
    PARSER.add_argument(
        "--scaler",
        default="MinMaxScaler")
    PARSER.add_argument(
        "output",
        type=utils.URLPath)
    configure_print_logging()
    main(PARSER.parse_args())
