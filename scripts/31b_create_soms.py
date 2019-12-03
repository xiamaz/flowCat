#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

import pandas as pd

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
    cases, _ = cases.filter_reasons(materials=flowcat.ALLOWED_MATERIALS, tubes=(1, 2))
    cases = cases.sample(1000)
    model = flowcat.som.CaseSingleSom.load(
        args.model, max_epochs=4, batch_size=50000, initial_radius=4, subsample_size=1000, tensorboard_dir=None)
    labels = []
    for case, res in model.transform_generator(cases):
        flowcat.som.save_som(res, args.output / case.id, save_config=False, subdirectory=False)
        labels.append({"label": case.id, "randnum": 0, "group": case.group})
    metadata = pd.DataFrame(labels)
    utils.save_csv(metadata, args.output + ".csv")
    utils.save_json(
        {
            "tubes": [model.tube],
            "dims": model.model.dims,
            "channels": model.model.markers,
        }, args.output + ".json")


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
