import json
from collections import defaultdict

from flowcat import utils, io_functions
from flowcat.dataset import case_dataset


def transform(
        data: utils.URLPath,
        meta: utils.URLPath,
        output: utils.URLPath,
        reference: utils.URLPath,
        transargs: json.loads = None,
        recreate: bool = False,
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
        transargs = {
            "max_epochs": 4,
            "batch_size": 50000,
            "initial_radius": 4,
            "end_radius": 1,
        }

    print(f"Loading referece from {reference}")
    model = io_functions.load_casesom(reference, **transargs)

    print(f"Trainsforming individual samples")
    output.mkdir()
    casesamples = defaultdict(list)
    count_samples = len(dataset) * len(model.models)
    countlen = len(str(count_samples))
    for i, (case, somsample) in enumerate(utils.time_generator_logger(model.transform_generator(dataset))):
        sompath = output / f"{case.id}_t{somsample.tube}.npy"
        io_functions.save_som(somsample.data, sompath, save_config=False)
        somsample.data = None
        somsample.path = sompath
        casesamples[case.id].append(somsample)
        print(f"[{str(i + 1).rjust(countlen, ' ')}/{count_samples}] Created tube {somsample.tube} for {case.id}")

    print(f"Saving result to new collection at {output}")
    som_dataset = case_dataset.CaseCollection([
        case.copy(samples=casesamples[case.id])
        for case in dataset
    ])
    som_dataset.selected_markers = {
        m.tube: m.model.markers for m in model.models.values()
    }
    io_functions.save_case_collection(som_dataset, output + ".json.gz")
    io_functions.save_json(
        {
            tube: {
                "dims": m.model.dims,
                "channels": m.model.markers,
            } for tube, m in model.models.items()
        }, output + "_config.json")
