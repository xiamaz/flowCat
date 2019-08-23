#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

import flowcat
from flowcat import utils, sommodels, io_functions
from flowcat.dataset.fcs import extract_name


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def train_model(dataset, markers=None, tensorboard=None, marker_name_only=False):
    """Create and train a SOM model using the given dataset."""
    if markers:
        selected_markers = io_functions.load_json(markers)
    else:
        selected_markers = dataset.selected_markers
        # modify marker names if marker_name_only
        if marker_name_only:
            selected_markers = {
                tube: [extract_name(marker) for marker in markers]
                for tube, markers in selected_markers.items()
            }

    # scaler = "StandardScaler"
    scaler = "MinMaxScaler"

    model = sommodels.casesom.CaseSom(
        tubes=selected_markers,
        tensorboard_dir=tensorboard,
        modelargs={
            "marker_name_only": marker_name_only,
            "max_epochs": 10,
            "batch_size": 50000,
            "initial_learning_rate": 0.05,  # default: 0.05
            "end_learning_rate": 0.01,  # default: 0.01
            "learning_cooling": "linear",
            "initial_radius": 24,
            "end_radius": 2,
            "radius_cooling": "linear",
            # "marker_images": sommodels.fcssom.MARKER_IMAGES_NAME_ONLY,
            "map_type": "toroid",
            "dims": (32, 32, -1),
            "scaler": scaler,
        })
    model.train(dataset)
    return model


def main(args):
    """Load case ids from json file to filter cases and train and save the created model."""
    output_dir = args.output

    dataset = io_functions.load_case_collection(args.data, args.meta)
    selected_labels = io_functions.load_json(args.cases)
    selected, _ = dataset.filter_reasons(labels=selected_labels)

    if args.tensorboard:
        tensorboard_dir = output_dir / "tensorboard"
    else:
        tensorboard_dir = None

    model = train_model(
        selected,
        markers=args.markers,
        tensorboard=tensorboard_dir,
        marker_name_only=args.marker_name_only)

    io_functions.save_casesom(model, output_dir)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    flowcat.parser.add_dataset_args(PARSER)
    PARSER.add_argument(
        "--tensorboard",
        help="Flag to enable tensorboard logging",
        action="store_true")
    PARSER.add_argument(
        "--markers",
        help="Input json file mapping tube number to markers",
        type=utils.URLPath)
    PARSER.add_argument(
        "--marker-name-only",
        help="Only use the marker name, not the antibody",
        action="store_true")
    PARSER.add_argument(
        "cases",
        type=utils.URLPath,
        help="Json file containing a number of case ids")
    PARSER.add_argument(
        "output",
        help="Output Reference SOM path",
        type=utils.URLPath)
    configure_print_logging()
    main(PARSER.parse_args())
