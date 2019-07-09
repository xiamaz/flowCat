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


def main(args):
    cases = flowcat.CaseCollection.from_path(args.input, metapath=args.meta)
    cases = cases.sample(1000)
    model = flowcat.som.CaseSingleSom.load(
        args.model, max_epochs=4, batch_size=50000, initial_radius=4, subsample_size=1000, tensorboard_dir=None)
    print(cases)
    print(cases.group_count)
    print(model)
    print(model.model.markers)
    for res in model.transform_generator(cases):
        print(res)
        flowcat.som.save_som(res, args.output / res.cases[0], save_config=False, subdirectory=False)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    PARSER.add_argument("-o", "--output", default=utils.URLPath("output"), type=utils.URLPath)
    PARSER.add_argument(
        "-i", "--input",
        default=utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F"),
        type=utils.URLPath)
    PARSER.add_argument("-m", "--meta", type=utils.URLPath)
    PARSER.add_argument(
        "model",
        type=utils.URLPath)
    PARSER.add_argument(
        "output",
        type=utils.URLPath)
    configure_print_logging()
    main(PARSER.parse_args())
