#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

import flowcat
from flowcat import utils


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def load_dataset(input_dir, metapath):
    return flowcat.CaseCollection.from_path(input_dir, metapath=metapath)


def main(args):
    dataset = load_dataset(args.input, args.meta)
    print(dataset)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    PARSER.add_argument("-o", "--output", default=utils.URLPath("output"), type=utils.URLPath)
    PARSER.add_argument(
        "-i", "--input",
        default=utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F"),
        type=utils.URLPath)
    PARSER.add_argument("-m", "--meta", type=utils.URLPath)
    configure_print_logging()
    main(PARSER.parse_args())
