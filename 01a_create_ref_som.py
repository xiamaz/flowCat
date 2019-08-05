#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

import flowcat
from flowcat import utils, som


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def main(args):
    output_dir = args.output

    dataset = flowcat.CaseCollection.from_path(args.input, metapath=args.meta)
    selected_labels = utils.load_json(args.cases)
    selected, _ = dataset.filter_reasons(labels=selected_labels)

    if args.markers:
        selected_markers = utils.load_json(args.markers)
    else:
        selected_markers = dataset.selected_markers

    if args.tensorboard:
        tensorboard_dir = output_dir / "tensorboard"
    else:
        tensorboard_dir = None

    model = som.CaseSom(
        tubes=selected_markers,
        materials=flowcat.ALLOWED_MATERIALS,
        modelargs={
            "marker_name_only": True,
            "max_epochs": 10,
            "batch_size": 10000,
            "marker_images": som.fcssom.MARKER_IMAGES_NAME_ONLY,
            "map_type": "toroid",
            "tensorboard_dir": tensorboard_dir,
            "dims": (32, 32, -1),
        })
    model.train(selected)
    model.save(output_dir / "model")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    PARSER.add_argument(
        "-i", "--input",
        desc="Input directory containing FCS files",
        type=utils.URLPath)
    PARSER.add_argument(
        "--tensorboard",
        desc="Flag to enable tensorboard logging",
        action="store_true")
    PARSER.add_argument(
        "--markers",
        desc="Input json file mapping tube number to markers",
        type=utils.URLPath)
    PARSER.add_argument(
        "-m", "--meta",
        desc="Metadata file for the dataset",
        type=utils.URLPath)
    PARSER.add_argument("cases", type=utils.URLPath)
    PARSER.add_argument("output", type=utils.URLPath)
    configure_print_logging()
    main(PARSER.parse_args())
