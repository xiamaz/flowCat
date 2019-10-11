#!/usr/bin/env python3
"""Move a given dataset including fcs from a given location to a new location."""
import json
from argmagic import argmagic
from flowcat import utils, io_functions


def move_dataset(
        meta: utils.URLPath,
        data: utils.URLPath,
        output: utils.URLPath,
        filters: json.loads = None,
        case_collection: bool = False):
    """
    Move a given dataset to a new location. The output location will contain:
    output/
        meta.json
        data

    Args:
        meta: Current data metadata.
        data: Current data fcs data.
        output: Destination directory to copy data to.
    """
    if case_collection:
        dataset = io_functions.load_case_collection_from_caseinfo(data, meta)
    else:
        dataset = io_functions.load_case_collection(data, meta)
    if filters:
        print(f"Filtering dataset using: {filters}")
        dataset = dataset.filter(**filters)
    print("Save dataset to", output)
    io_functions.save_case_collection_with_data(dataset, output)


if __name__ == "__main__":
    argmagic(move_dataset)
