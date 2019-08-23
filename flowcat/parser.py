# pylint: skip-file
# flake8: noqa
"""
CLI Interface components.
"""
from argparse import ArgumentParser, Namespace
from flowcat.utils import URLPath
from flowcat import CaseCollection
from flowcat.dataset import case_dataset


def add_dataset_args(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments for specifying a defined case dataset.

    An input and a meta instance will be added.

    Args:
        parser: Parser instance.

    Returns:
        Parser with dataset options added as an argument group.
    """

    dataset_group = parser.add_argument_group("Dataset options")
    dataset_group.add_argument(
        "-i", "--input",
        type=URLPath,
        required=True,
        help="Path to dataset",
    )
    dataset_group.add_argument(
        "-m", "--meta",
        type=URLPath,
        required=True,
        help="Path to dataset metadata. Do not include file ending.")

    return parser


def get_dataset(args: Namespace) -> CaseCollection:
    cases = case_dataset.load_case_collection_from_caseinfo(args.input, args.meta)
    return cases
