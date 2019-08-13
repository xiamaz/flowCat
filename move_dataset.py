#!/usr/bin/env python3
"""Move a given dataset including fcs from a given location to a new location."""
import shutil

import argmagic
import flowcat
from flowcat.utils import URLPath, load_json


def move_dataset(meta: URLPath, data: URLPath, labels: URLPath, output: URLPath):
    """
    Move a given dataset to a new location. The output location will contain:
    output/
        metadata.json
        fcsdata

    Args:
        meta: Current data metadata.
        data: Current data fcs data.
        output: Destination directory to copy data to.
    """
    output_fcs_path = output / "fcsdata"
    output_fcs_path.mkdir()
    dataset = flowcat.CaseCollection.load(inputpath=data, metapath=meta)

    case_labels = load_json(labels)
    dataset, _ = dataset.filter_reasons(labels=case_labels)
    print(dataset)

    for case in dataset:
        for tsample in case.filepaths:
            cur_path = data / tsample.path
            new_path = output_fcs_path / tsample.path.name
            shutil.copyfile(str(cur_path), str(new_path))
            tsample.path = new_path
    dataset.save(output / "metadata")


if __name__ == "__main__":
    argmagic.argmagic(move_dataset)
