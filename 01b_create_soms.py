#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

import pandas as pd

import flowcat
from flowcat import utils, io_functions
from flowcat.dataset import sample


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


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
    for case, res in utils.time_generator_logger(model.transform_generator(dataset)):
        io_functions.save_som(res, output / case.id, save_config=False, subdirectory=False)

    somcases = []
    for case in dataset:
        somsamples = []
        for tube, tmodel in model.models.items():
            som_id = f"{case.id}_t{tube}_{tmodel.run_identifier}"
            fcs_sample = case.get_tube(tube)
            somsamples.append(sample.SOMSample(
                id=som_id,
                case_id=case.id,
                path=output / f"{case.id}_t{tube}.csv",
                original_id=fcs_sample.id,
                date=tmodel.model_time,
                tube=tube,
                dims=tmodel.model.dims,
                markers=tmodel.model.markers,
            ))
        somcases.append(case.copy(samples=somsamples))

    somcollection = flowcat.dataset.case_dataset.CaseCollection(somcases)
    io_functions.save_json(somcollection, output + "_collection.json")

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
        }, output + ".json")


def main(args):
    """Load a model with given transforming arguments and transform individual
    cases."""
    cases = io_functions.load_case_collection(args.data, args.meta)
    # cases = cases.sample(1, groups=["CLL", "normal"])

    if args.tensorboard:
        tensorboard_dir = args.output / "tensorboard"
    else:
        tensorboard_dir = None

    # Training parameters for the model can be respecified, the only difference
    # between transform and normal traninig, is that after a transformation is
    # completed, the original weights will be restored to the model.
    model = io_functions.load_casesom(
        args.model,
        # marker_images=flowcat.sommodels.fcssom.MARKER_IMAGES_NAME_ONLY,
        max_epochs=4,
        initial_learning_rate=0.05,
        end_learning_rate=0.01,
        batch_size=50000,
        initial_radius=4,
        end_radius=1,
        # subsample_size=1000,
        tensorboard_dir=tensorboard_dir)

    transform_cases(cases, model, args.output)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    PARSER = flowcat.parser.add_dataset_args(PARSER)
    PARSER.add_argument(
        "--tensorboard",
        help="Flag to enable tensorboard logging",
        action="store_true")
    PARSER.add_argument(
        "model",
        type=utils.URLPath)
    PARSER.add_argument(
        "output",
        type=utils.URLPath)
    configure_print_logging()
    main(PARSER.parse_args())
