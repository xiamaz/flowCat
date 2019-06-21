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
    output_dir = args.output / args.name

    dataset = flowcat.CaseCollection.from_path(args.input, metapath=args.meta)
    normal_labels = utils.load_json("data/normals.json")
    normals, _ = dataset.filter_reasons(labels=normal_labels)
    print(normals.labels)

    # TODO: Generate a SOM for all tubes for the given labels.
    # Visualize using tensorboard
    # Save everything into a single, folder which we can use in the next script
    # to create single SOMs


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    PARSER.add_argument("-o", "--output", default=utils.URLPath("output/01a-ref-som"), type=utils.URLPath)
    PARSER.add_argument(
        "-i", "--input",
        default=utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F"),
        type=utils.URLPath)
    PARSER.add_argument("-m", "--meta", type=utils.URLPath)
    PARSER.add_argument("name")
    configure_print_logging()
    main(PARSER.parse_args())
