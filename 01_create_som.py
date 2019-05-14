#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

from flowcat import utils, configuration, mappings, som
from flowcat.dataset import case_dataset


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


def create_new_reference(args):
    """Create a new reference SOM."""
    config = args.refconfig
    print(f"Creating reference SOM with name {config('name')}")

    casespath = utils.get_path(config("dataset", "names", "FCS"), args.pathconfig("input", "FCS"))
    cases = case_dataset.CaseCollection.from_path(casespath)
    labels = utils.load_labels(config("dataset", "labels"))
    data = cases.filter(labels=labels, **config("dataset", "filters"))

    if args.tensorboard:
        tensorboard_path = args.tensorboard / args.name
        print(f"Creating tensorboard logs in {tensorboard_path}")
    else:
        tensorboard_path = None

    # load reference if available
    reference = config("reference")
    if reference is not None:
        reference = som.load_som(reference, config("dataset", "filters", "tubes"), suffix=False)

    return som.create_som(data, config, tensorboard_path, reference=reference)


def generate_reference(args):
    """Generate a reference SOMmap using the given configuration."""
    # load existing if it already exists
    path = utils.URLPath(args.pathconfig("output", "som-reference"), args.refconfig("name"))

    if path.exists():
        print(f"Loading existing references in {path}")
        tubes = args.refconfig("dataset", "filters", "tubes")
        return som.load_som(path, tubes, suffix=False)

    data = create_new_reference(args)
    print(f"Saving reference SOM in {path}")
    som.save_som_dict(data, path, suffix=False)
    # Save reference configuration
    args.refconfig.to_file(som.get_config_path(path))

    return data


def generate_soms(args):
    config = args.indivconfig
    output_dir = utils.URLPath(args.pathconfig("output", "som-sample"), config("name"))
    print(f"Create individual SOM in {output_dir}")

    if config("reference"):
        reference = generate_reference(args)
    else:
        reference = None

    data = som.create_filtered_data(config, pathconfig=args.pathconfig)

    metadata = som.create_indiv_soms(
        data, config, output_dir, reference=reference,
        tensorboard_dir=args.tensorboard, pathconfig=args.pathconfig)

    utils.save_csv(metadata, output_dir + ".csv")

    config.to_file(output_dir / "config.toml")


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
        choices=["refconfig", "indivconfig", "pathconfig"],
        help="Generate reference or individual SOM config",
        default="refconfig")
    parser_conf.add_argument(
        "-o", "--output",
        help="Output path to save configuration",
        default="")
    parser_conf.set_defaults(fun=run_config)

    parser_create = subparsers.add_parser(
        "som", help="Generate individual SOM, will also create reference if missing")
    parser_create.add_argument(
        "--no-recreate-samples", help="Do not regenerate already created individual samples", action="store_true")
    parser_create.set_defaults(fun=generate_soms)

    parser_ref = subparsers.add_parser("reference", help="Generate reference SOM")
    parser_ref.set_defaults(fun=generate_reference)

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
    if args.indivconfig:
        args.indivconfig = som.IndivConfig.from_file(args.indivconfig)
    else:
        args.indivconfig = som.IndivConfig.generate_config(args)

    if hasattr(args, "fun"):
        args.fun(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
