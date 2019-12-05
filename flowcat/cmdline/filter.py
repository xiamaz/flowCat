import json
from flowcat import utils, io_functions


def filter(
        data: utils.URLPath,
        filters: json.loads,
        output: utils.URLPath = None,
        meta: utils.URLPath = None,
        sample: int = 0,
        move_samples: bool = False,
):
    """Filter data on the given filters and output resulting dataset metadata
    to destination.

    Args:
        data: Path to fcs data.
        meta: Path to dataset metadata.
        output: Path to output for metadata.
        filters: Filters for individual cases.
        sample: Number of cases per group.
        move_samples: Destination will also include sample data.
    """
    print(f"Loading existing dataset from {data} with metadata in {meta}")
    try:
        dataset = io_functions.load_case_collection(data, meta)
    except TypeError:
        dataset = io_functions.load_case_collection_from_caseinfo(data, meta)

    dataset = dataset.filter(**filters)
    if sample:
        dataset = dataset.sample(sample)

    print(f"Filtering down to {dataset}")
    print(dataset.group_count)

    if output:
        print("Saving", dataset, f"to {output}")
        if move_samples:
            io_functions.save_case_collection_with_data(dataset, output)
        else:
            io_functions.save_case_collection(dataset, output)
