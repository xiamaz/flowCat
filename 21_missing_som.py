# Sanely handle missing values in SOM
import logging
import numpy as np
import sklearn as sk
import flowcat


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        flowcat.utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    flowcat.utils.add_logger(rootname, handlers, level=logging.DEBUG)


def train_native():
    ds_missing = flowcat.CaseCollection.from_path("output/missing")
    ds_subsample = flowcat.CaseCollection.from_path("output/subsample")

    tube = 1
    markers = ds_missing.get_markers(tube)
    markers_list = markers.index.values

    filepath = ds_missing.data[0].get_tube(tube)
    fcsdata = filepath.data

    sample_size = 5000

    model = flowcat.models.FCSSom(
        (32, 32, -1), markers=markers_list,
        batch_size=500,
        tensorboard_dir="output/21-tensorboard/native")
    model.train([fcsdata], sample=sample_size)


if __name__ == "__main__":
    configure_print_logging()
    train_native()
