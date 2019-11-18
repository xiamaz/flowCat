from flowcat import utils, io_functions


def dataset(data: utils.URLPath, meta: utils.URLPath):
    """Print information on the given dataset."""
    try:
        dataset = io_functions.load_case_collection(data, meta)
    except TypeError:
        dataset = io_functions.load_case_collection_from_caseinfo(data, meta)

    print(f"Loaded dataset from {meta}", dataset)
    print(dataset.group_count)
