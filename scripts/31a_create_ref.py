#!/usr/bin/env python3
"""
Create SOM maps to be used as references and individual maps to be used for
visualization and classification.
"""
import argparse
import logging

import flowcat
from flowcat import utils, som, io_functions


LOGGER = logging.getLogger(__name__)


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger(rootname, handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def main(args):
    output_dir = args.output / args.name

    dataset = flowcat.CaseCollection.from_path(args.input, metapath=args.meta)
    selected_labels = io_functions.load_json("data/selected_cases.json")
    selected, _ = dataset.filter_reasons(labels=selected_labels)
    selected = dataset.sample(count=1, groups=["CLL", "normal"])
    print(selected.labels)

    joined_tubes = io_functions.load_json("output/00-dataset-test/munich_bonn_tubes.json")
    print(joined_tubes)

    # TODO: Generate a SOM for all tubes for the given labels.
    # Visualize using tensorboard
    # Save everything into a single, folder which we can use in the next script
    # to create single SOMs
    model = som.CaseSingleSom(
        tube=1,
        materials=flowcat.ALLOWED_MATERIALS,
        markers=joined_tubes["1"],
        marker_name_only=True,
        max_epochs=10,
        batch_size=10000,
        marker_images=som.fcssom.MARKER_IMAGES_NAME_ONLY,
        map_type="toroid",
        tensorboard_dir=output_dir / "tensorboard",
        dims=(32, 32, -1)
    )
    model.train(selected)
    model.save(output_dir / "model")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(usage="Generate references and individual SOMs.")
    PARSER.add_argument("-o", "--output", default=utils.URLPath("output/01a-ref-som"), type=utils.URLPath)
    PARSER.add_argument(
        "-i", "--input",
        default=utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F"),
        type=utils.URLPath)
    PARSER.add_argument("-m", "--meta", type=utils.URLPath)
    PARSER.add_argument("name")
    configure_print_logging()
    main(PARSER.parse_args())
