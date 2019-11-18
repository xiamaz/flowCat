from flowcat import utils, io_functions


def dataset(data: utils.URLPath, meta: utils.URLPath):
    """Print information on the given dataset."""
    data = utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F")
    try:
        dataset = io_functions.load_case_collection(data, meta)
    except TypeError:
        dataset = io_functions.load_case_collection_from_caseinfo(data, meta)

    train = io_functions.load_case_collection(data, meta)
    test = io_functions.load_case_collection(data, meta_test)
    print(f"Loaded dataset from {meta}", dataset)
    print(dataset.group_count)
