#!/usr/bin/env python3
import sys
import math
import os
import logging
import argparse

from flowcat import utils, classify, configuration


LOGGER = logging.getLogger(__name__)


def setup_logging_from_args(args):
    classification_path = args.pathconfig("output", "classification")
    outpath = utils.URLPath(f"{classification_path}/{config('name')}")
    logpath = outpath / f"classification.log"
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


def config(args):
    setup_logging(filelog=None, printlevel=logging.ERROR)
    config = getattr(args, args.type)

    if args.output is None:
        print("HINT: Redirect stderr to get config output suitable for piping.", file=sys.stderr)
        print(config)
    else:
        print(f"Writing configuration to {args.output}")
        config.to_file(args.output)


def main():
    parser = argparse.ArgumentParser(description="Classify samples")
    configuration.PathConfig.add_to_arguments(parser)
    classify.SOMClassifierConfig.add_to_arguments(parser)
    subparsers = parser.add_subparsers()

    # Save configuration files in specified in the create configuration file
    cparser = subparsers.add_parser(
        "config",
        help="Output the current configuration. Hint: Get rid of import messages by eg piping stderr to /dev/null")
    cparser.add_argument(
        "-o", "--output",
        help="Write configuration to output file.",
        type=utils.URLPath)
    cparser.add_argument(
        "--type",
        help="Output pathconfig instead of model configuration.",
        choices=["pathconfig", "modelconfig"],
        default="modelconfig",
    )
    cparser.set_defaults(fun=config)

    # Run the classification process
    rparser = subparsers.add_parser("run", help="Run classification")
    rparser.add_argument(
        "--seed",
        help="Seed for random number generator",
        type=int)
    rparser.add_argument(
        "--model",
        help="Use an existing model",
        type=utils.URLPath) # TODO
    rparser.add_argument(
        "--name",
        help="Set an alternative output name.")
    rparser.add_argument(
        "-v", "--verbose",
        help="Control verbosity. -v is info, -vv is debug",
        action="count")
    rparser.set_defaults(fun=classify.run)

    args = parser.parse_args()

    if not hasattr(args, "fun") or args.fun is None:
        parser.print_help()
    else:
        setup_logging_from_args(args)
        args.fun(args)


if __name__ == "__main__":
    main()
