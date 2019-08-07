#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

import flowcat
from flowcat import utils, som
from flowcat.dataset.fcs import extract_name


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def train_model(dataset, markers=None, tensorboard=None):
    """Create and train a SOM model using the given dataset."""
    marker_name_only = True
    if markers:
        selected_markers = utils.load_json(markers)
    else:
        selected_markers = dataset.selected_markers
        # modify marker names if marker_name_only
        selected_markers = {
            tube: [extract_name(marker) for marker in markers]
            for tube, markers in selected_markers.items()
        }

    model = som.CaseSom(
        tubes=selected_markers,
        tensorboard_dir=tensorboard,
        modelargs={
            "marker_name_only": marker_name_only,
            "max_epochs": 10,
            "batch_size": 10000,
            "marker_images": som.fcssom.MARKER_IMAGES_NAME_ONLY,
            "map_type": "toroid",
            "dims": (32, 32, -1),
        })
    model.train(dataset)
    return model


def main(args):
    """Load case ids from json file to filter cases and train and save the created model."""
    output_dir = args.output

    dataset = flowcat.parser.get_dataset(args)
    selected_labels = utils.load_json(args.cases)
    selected, _ = dataset.filter_reasons(labels=selected_labels)

    if args.tensorboard:
        tensorboard_dir = output_dir / "tensorboard"
    else:
        tensorboard_dir = None

    model = train_model(selected, markers=args.markers, tensorboard=tensorboard_dir)

    model.save(output_dir)


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
        "cases",
        type=utils.URLPath,
        help="Json file containing a number of case ids")
    PARSER.add_argument(
        "output",
        help="Output Reference SOM path",
        type=utils.URLPath)
    configure_print_logging()
    main(PARSER.parse_args())
