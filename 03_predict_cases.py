#!/usr/bin/env python3
import sys
import math
import os
import logging
import argparse

from flowcat import utils, classify, configuration
from flowcat.dataset import case_dataset


LOGGER = logging.getLogger(__name__)


def setup_logging_from_args(args):
    logpath = args.logpath
    if logpath:
        logpath = utils.URLPath(logpath)
        logpath.local.parent.mkdir(parents=True, exist_ok=True)
    return setup_logging(logpath, printlevel=args_loglevel(args.verbose))


def setup_logging(filelog=None, filelevel=logging.DEBUG, printlevel=logging.WARNING):
    """Setup logging to both visible output and file output.
    Args:
        filelog: Logging file. Will not log to file if None
        filelevel: Logging level inside file.
        printlevel: Logging level for visible output.
    """
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=printlevel),
    ]
    if filelog is not None:
        handlers.append(
            utils.create_handler(logging.FileHandler(str(filelog)), level=filelevel)
        )

    utils.add_logger("flowcat", handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def args_loglevel(vlevel):
    """Get logging level from number of verbosity chars."""
    if not vlevel:
        return logging.WARNING
    if vlevel == 1:
        return logging.INFO
    return logging.DEBUG


def run(args):
    modelpath = args.modelpath
    assert modelpath.exists(), f"{modelpath} not found"

    LOGGER.info("Loading existing model at %s", modelpath)
    model, transform, groups, filters = classify.load_model(modelpath)

    casepath = args.cases
    labelpath = args.labels
    labels = utils.load_labels(labelpath)
    LOGGER.info("Loading %d labels in %s", len(labels), labelpath)
    path = utils.get_path(casepath, args.pathconfig("input", "FCS"))
    LOGGER.info("Loading cases in %s", casepath)

    cases = case_dataset.CaseCollection.from_path(path)
    LOGGER.info("Additional filters: %s", filters)
    filtered = cases.filter(labels=labels, **filters)

    indata = transform(filtered)
    predictions = classify.predict(model, indata, groups, filtered.labels)

    utils.save_csv(predictions, args.output)


def main():
    parser = argparse.ArgumentParser(description="Predict cases")
    configuration.PathConfig.add_to_arguments(parser)
    parser.add_argument(
        "-v", "--verbose",
        help="Control verbosity. -v is info, -vv is debug",
        action="count")
    parser.add_argument(
        "--logpath", help="Optionally log to location")
    parser.add_argument(
        "--seed",
        help="Seed for random number generator",  # TODO use seed
        type=int)
    parser.add_argument(
        "--cases",
        default="decCLL-9F",
        help="Cases to be predicted.")
    parser.add_argument(
        "--output",
        help="Output path.",
        default="predictions.csv",
        type=utils.URLPath)
    parser.add_argument(
        "modelpath",
        help="Path to trained model directory",
        type=utils.URLPath)
    parser.add_argument(
        "labels",
        help="List of labels in a json file.",
        type=utils.URLPath)
    args = parser.parse_args()

    setup_logging_from_args(args)
    run(args)


if __name__ == "__main__":
    main()
