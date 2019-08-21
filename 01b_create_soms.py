#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

import pandas as pd

import flowcat
from flowcat import utils


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
    labels = []
    for case, res in utils.time_generator_logger(model.transform_generator(dataset)):
        flowcat.som.save_som(res, output / case.id, save_config=False, subdirectory=False)
        labels.append({"label": case.id, "randnum": 0, "group": case.group})

    # Save metadata into an additional csv file with the same name
    metadata = pd.DataFrame(labels)
    utils.save_csv(metadata, output + ".csv")
    utils.save_json(
        {
            tube: {
                "dims": m.model.dims,
                "channels": m.model.markers,
            } for tube, m in model.models.items()
        }, output + ".json")


def main(args):
    """Load a model with given transforming arguments and transform individual
    cases."""
    cases = flowcat.parser.get_dataset(args)

    if args.tensorboard:
        tensorboard_dir = args.output / "tensorboard"
    else:
        tensorboard_dir = None

    # Training parameters for the model can be respecified, the only difference
    # between transform and normal traninig, is that after a transformation is
    # completed, the original weights will be restored to the model.
    model = flowcat.som.CaseSom.load(
        args.model,
        marker_images=flowcat.som.fcssom.MARKER_IMAGES_NAME_ONLY,
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
