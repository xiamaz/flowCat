import json

from flowcat import utils, io_functions
from argmagic import argmagic


def filter(
        data: utils.URLPath,
        meta: utils.URLPath,
        output: utils.URLPath,
        filters: json.loads,
        sample: int = 0):
    """Filter data on the given filters and output resulting dataset metadata
    to destination.

    Args:
        data: Path to fcs data.
        meta: Path to dataset metadata.
        output: Path to output for metadata.
        filters: Filters for individual cases.
        sample: Number of cases per group.
    """
    try:
        dataset = io_functions.load_case_collection(data, meta)
    except TypeError:
        dataset = io_functions.load_case_collection_from_caseinfo(data, meta)
    dataset = dataset.filter(**filters)
    if sample:
        dataset = dataset.sample(sample)
    print("Saving", dataset)
    io_functions.save_case_collection(dataset, output)


def train(data: utils.URLPath, output: utils.URLPath):
    """Train a new model from the given data."""


def predict(case, model: utils.URLPath, output: utils.URLPath):
    """Generate predictions and plots for a single case.

    Args:
        case: Single case with FCS files.
        model: Path to model containing CNN and SOMs.
        output: Destination for plotting.
    """


def main():
    argmagic([predict, train, filter])
