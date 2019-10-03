#!/usr/bin/env python3
"""
Train a reference SOM and optionally use that to transform the given
dataset.
"""
import logging
import json
from collections import defaultdict

import pandas as pd
from argmagic import argmagic

from flowcat import utils, io_functions, sommodels
from flowcat.dataset import case_dataset
from flowcat.dataset.fcs import extract_name


LOGGER = logging.getLogger(__name__)


def transform_cases(dataset, model, output):
    """Create individidual SOMs for all cases in the dataset.
    Args:
        dataset: CaseIterable with a number of cases, for which SOMs should be
                 generated.
        model: Model with initial weights, which should be used for generation
               of SOMs.
        output: Output directory for SOMs

    Returns:
        Nothing.
    """
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

    labels = [{"label": case.id, "randnum": 0, "group": case.group} for case in dataset]
    # Save metadata into an additional csv file with the same name
    metadata = pd.DataFrame(labels)
    io_functions.save_csv(metadata, output + ".csv")
    io_functions.save_json(
        {
            tube: {
                "dims": m.model.dims,
                "channels": m.model.markers,
            } for tube, m in model.models.items()
        }, output + "_config.json")


def train_model(
        dataset,
        markers=None,
        tensorboard=None,
        modelargs=None,
) -> sommodels.casesom.CaseSom:
    """Create and train a SOM model using the given dataset."""
    if modelargs is None:
        modelargs = {
            "marker_name_only": False,
            "max_epochs": 10,
            "batch_size": 50000,
            "initial_radius": 16,
            "end_radius": 2,
            "radius_cooling": "linear",
            # "marker_images": sommodels.fcssom.MARKER_IMAGES_NAME_ONLY,
            "map_type": "toroid",
            "dims": (32, 32, -1),
            "scaler": "MinMaxScaler",
        }

    if markers:
        selected_markers = io_functions.load_json(markers)
    else:
        selected_markers = dataset.selected_markers
        # modify marker names if marker_name_only
        if modelargs.get("marker_name_only", False):
            selected_markers = {
                tube: [extract_name(marker) for marker in markers]
                for tube, markers in selected_markers.items()
            }

    model = sommodels.casesom.CaseSom(
        tubes=selected_markers,
        tensorboard_dir=tensorboard,
        modelargs=modelargs,
    )
    model.train(dataset)
    return model


def main(
        data: utils.URLPath,
        meta: utils.URLPath,
        output: utils.URLPath,
        reference_ids: utils.URLPath = None,
        reference: utils.URLPath = None,
        tensorboard_dir: utils.URLPath = None,
        modelargs: json.loads = None,
        transargs: json.loads = None,
        mode: str = "fit_transform",
):
    """
    Train a SOM and use its weights to initialize individual SOM training.

    Args:
        data: Path to fcs data.
        meta: Path to dataset metadata, this should correctly reference fcs data.
        output: Path to output model and transformed cases.
        reference_ids: Optionally list ids to be used for reference SOM generation.
        reference: Optionally use pretrained model.
        modelargs: Optionally give specific options for reference SOM generation.
        transargs: Optionally give specific options for transforming individual SOMs.
        mode: Whether to fit or to transform. Default both.
    """
    dataset = io_functions.load_case_collection(data, meta)

    if reference is None:
        reference = train_model(dataset, modelargs=modelargs)
        reference_output = output / "reference"
        io_functions.save_casesom(reference, reference_output)
        reference = reference_output

    if transargs is None:
        transargs = {
            "max_epochs": 4,
            "batch_size": 50000,
            "initial_radius": 4,
            "end_radius": 1,
        }

    model = io_functions.load_casesom(
        reference,
        tensorboard_dir=tensorboard_dir,
        **transargs
    )

    som_output = output / "som"
    transform_cases(dataset, model, som_output)


if __name__ == "__main__":
    argmagic(main)
