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


def train_native(sample, outname):
    fcsdata = sample.data

    sample_size = 5000

    model = flowcat.models.FCSSom(
        (32, 32, -1), markers=sample.markers,
        batch_size=500, tube=sample.tube,
        tensorboard_dir=f"output/21-tensorboard/{outname}")
    model.train([fcsdata], sample=sample_size)

    weights = model.weights

    flowcat.save_som(weights, f"output/21-tfsom/{outname}")


if __name__ == "__main__":
    configure_print_logging()

    ds_missing = flowcat.CaseCollection.from_path("output/missing")
    sample_missing = ds_missing.data[0].get_tube(1)
    train_native(sample_missing, "native-missing")

    ds_subsample = flowcat.CaseCollection.from_path("output/subsample")
    sample_subsample = ds_subsample.data[0].get_tube(1)
    train_native(sample_subsample, "native-subsample")
