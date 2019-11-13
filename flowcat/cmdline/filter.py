import json
from flowcat import utils, io_functions


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
