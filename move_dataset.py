#!/usr/bin/env python3
"""Move a given dataset including fcs from a given location to a new location."""
from typing import List
import json
from argmagic import argmagic
from flowcat import utils, io_functions
from flowcat.dataset.case_dataset import CaseCollection


def lenient_load_collection(data, meta):
    try:
        dataset = io_functions.load_case_collection(data, meta)
    except TypeError:
        dataset = io_functions.load_case_collection_from_caseinfo(data, meta)
    return dataset


def merge(
        metas: List[utils.URLPath],
        datas: List[utils.URLPath],
        output: utils.URLPath,
        filters: json.loads = None,
):
    """Merges the given list of datasets into a single dataset and output it
    into the given output directory."""
    print("Loading datasets")
    datasets = list(map(lambda t: lenient_load_collection(*t), zip(datas, metas)))
    print("Filtering datasets individually.")
    if filters:
        datasets = list(map(lambda d: d.filter(**filters), datasets))

    # merge datasets and check for potential conflicts
    print("Checking for duplicates in datasets")
    for dataset in datasets:
        labels = dataset.labels
        for other_dataset in datasets:
            if other_dataset.meta_path == dataset.meta_path:
                continue
            for label in other_dataset.labels:
                if label in labels:
                    raise RuntimeError(f"Duplicate label {label} in {dataset} and {other_dataset}")

    # move data first individually and then merge manually
    dataset = CaseCollection([c for d in datasets for c in d])
    print(f"Moving merged dataset to {output}")
    io_functions.save_case_collection_with_data(dataset, output)


def move(
        meta: utils.URLPath,
        data: utils.URLPath,
        output: utils.URLPath,
        filters: json.loads = None):
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
    dataset = lenient_load_collection(data, meta)
    if filters:
        print(f"Filtering dataset using: {filters}")
        dataset = dataset.filter(**filters)
    print("Save dataset to", output)
    io_functions.save_case_collection_with_data(dataset, output)


if __name__ == "__main__":
    argmagic([move, merge])
