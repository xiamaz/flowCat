"""Test retraining procedure with many different settings and plot them."""
# pylint: skip-file
# flake8: noqa
from collections import defaultdict
from argmagic import argmagic

from flowcat import utils, io_functions
from flowcat.dataset import case_dataset


def transform_data(dataset, model, output):
    output.mkdir()

    casesamples = defaultdict(list)
    for case, somsample in utils.time_generator_logger(model.transform_generator(dataset)):
        sompath = output / f"{case.id}_t{somsample.tube}.npy"
        io_functions.save_som(somsample.data, sompath, save_config=False)
        somsample.data = None
        somsample.path = sompath
        casesamples[case.id].append(somsample)

    somcases = []
    for case in dataset:
        somcases.append(case.copy(samples=casesamples[case.id]))

    somcollection = case_dataset.CaseCollection(somcases)
    io_functions.save_json(somcollection, output + ".json")
    io_functions.save_json(
        {
            tube: {
                "dims": m.model.dims,
                "channels": m.model.markers,
            } for tube, m in model.models.items()
        }, output + "_config.json")


def main(data: utils.URLPath, reference: utils.URLPath, output: utils.URLPath):
    """
    """
    cases = io_functions.load_case_collection(data, data / data.name + ".json")
    default_settings = {
        "max_epochs": 4,
        "initial_learning_rate": 0.05,
        "end_learning_rate": 0.01,
        "batch_size": 50000,
        "initial_radius": 4,
        "end_radius": 1,
    }
    # settings = [
    #     ("learning_rate_001_0001", {"initial_learning_rate": 0.01, "end_learning_rate": 0.001}),
    #     ("learning_rate_001_001", {"initial_learning_rate": 0.01, "end_learning_rate": 0.01}),
    #     ("learning_rate_005_0001", {"initial_learning_rate": 0.05, "end_learning_rate": 0.001}),
    #     ("learning_rate_005_001", {"initial_learning_rate": 0.05, "end_learning_rate": 0.01}),
    #     ("learning_rate_005_005", {"initial_learning_rate": 0.05, "end_learning_rate": 0.05}),
    #     ("learning_rate_05_0001", {"initial_learning_rate": 0.5, "end_learning_rate": 0.001}),
    #     ("learning_rate_05_001", {"initial_learning_rate": 0.5, "end_learning_rate": 0.01}),
    #     ("learning_rate_05_01", {"initial_learning_rate": 0.5, "end_learning_rate": 0.1}),
    #     ("learning_rate_05_05", {"initial_learning_rate": 0.5, "end_learning_rate": 0.5}),
    # ]
    settings = [
        ("radius_24_1", {"initial_radius": 24, "end_radius": 1}),
        ("radius_24_2", {"initial_radius": 24, "end_radius": 2}),
        ("radius_24_1", {"initial_radius": 16, "end_radius": 1}),
        ("radius_16_2", {"initial_radius": 16, "end_radius": 2}),
        ("radius_8_1", {"initial_radius": 8, "end_radius": 1}),
        ("radius_8_2", {"initial_radius": 8, "end_radius": 2}),
        ("radius_4_1", {"initial_radius": 4, "end_radius": 1}),
        ("radius_4_2", {"initial_radius": 4, "end_radius": 2}),
    ]
    for name, setting in settings:
        model = io_functions.load_casesom(
            reference,
            **{**default_settings, **setting},
        )
        transform_data(cases, model, output / name)


argmagic(main)
