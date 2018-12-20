#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

from flowcat import utils, configuration, mappings, som


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def run_config(args):
    config = getattr(args, args.type)
    if args.output:
        config.to_file(utils.URLPath(args.output))
    else:
        print(config)


def run_groups(args):
    """Create stuff for each group."""
    tbpath = args.tensorboard
    for group in mappings.GROUPS:
        args.refconfig.data["dataset"]["filters"]["groups"] = [group]
        args.refconfig.data["name"] = f"single_{group}"
        args.tensorboard = tbpath / group
        som.generate_reference(args)


def run_infiltration(args):
    """Create single runs with cases with different levels of infiltration."""
    tbpath = args.tensorboard
    args.refconfig.data["dataset"]["labels"] = None
    args.refconfig.data["dataset"]["filters"]["groups"] = [args.group]
    for infiltration in range(10, 101, 10):
        print(f"Creating a reference SOM using infiltration in range {infiltration - 10} to {infiltration}")
        args.refconfig.data["name"] = f"{args.group}_i{infiltration}"
        args.tensorboard = tbpath / f"{args.group}_i{infiltration}"
        args.refconfig.data["dataset"]["filters"]["infiltration_max"] = infiltration
        args.refconfig.data["dataset"]["filters"]["infiltration"] = infiltration - 10
        som.generate_reference(args)


def main():
    configure_print_logging()

    parser = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    som.ReferenceConfig.add_to_arguments(parser, conv=str)
    som.IndivConfig.add_to_arguments(parser, conv=str)
    configuration.PathConfig.add_to_arguments(parser)
    parser.add_argument(
        "--tensorboard",
        help="Tensorboard directory",
        type=utils.URLPath)
    parser.add_argument(
        "--name",
        help="Name of the current run. Will be used as output folder name.",
        type=str, default="testname")
    subparsers = parser.add_subparsers()

    parser_conf = subparsers.add_parser("config", help="Generate the config to a specified directory")
    parser_conf.add_argument(
        "--type",
        choices=["refconfig", "somconfig", "pathconfig"],
        help="Generate reference or individual SOM config",
        default="refconfig")
    parser_conf.add_argument(
        "-o", "--output",
        help="Output path to save configuration",
        default="")
    parser_conf.set_defaults(fun=run_config)

    parser_create = subparsers.add_parser("som", help="Generate individual SOM, will also create reference if missing")
    parser_create.add_argument(
        "--no-recreate-samples", help="Do not regenerate already created individual samples", action="store_true")
    parser_create.set_defaults(fun=som.generate_soms)

    parser_ref = subparsers.add_parser("reference", help="Generate reference SOM")
    parser_ref.set_defaults(fun=som.generate_reference)

    parser_each = subparsers.add_parser("group", help="Create ref for each group.")
    parser_each.set_defaults(fun=run_groups)

    parser_infil = subparsers.add_parser("infiltration", help="Create ref for infiltration steps.")
    parser_infil.add_argument("--group", default="CLL")
    parser_infil.set_defaults(fun=run_infiltration)

    args = parser.parse_args()

    if args.refconfig:
        args.refconfig = som.ReferenceConfig.from_file(args.refconfig)
    else:
        args.refconfig = som.ReferenceConfig.generate_config(args)
    if args.somconfig:
        args.somconfig = som.IndivConfig.from_file(args.somconfig)
    else:
        args.somconfig = som.IndivConfig.generate_config(args)

    if hasattr(args, "fun"):
        args.fun(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
