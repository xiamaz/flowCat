import json

from flowcat import utils, io_functions
from flowcat.constants import DEFAULT_TRANSFORM_SOM_ARGS
from flowcat.flowcat import transform_dataset_to_som


def transform(
        data: utils.URLPath,
        meta: utils.URLPath,
        output: utils.URLPath,
        reference: utils.URLPath,
        transargs: json.loads = None,
        sample: int = 0):
    """Transform dataset using a reference SOM.

    Args:
        recreate: Delete and recreate SOMs even if they already exist.
        sample: Number of samples to transform from each group, only useful for testing purposes.
    """
    dataset = io_functions.load_case_collection(data, meta)

    # randomly sample 'sample' number cases from each group
    if sample:
        dataset = dataset.sample(sample)

    if transargs is None:
        transargs = DEFAULT_TRANSFORM_SOM_ARGS

    print(f"Loading referece from {reference}")
    model = io_functions.load_casesom(reference, **transargs)

    transform_dataset_to_som(model, dataset, output)
