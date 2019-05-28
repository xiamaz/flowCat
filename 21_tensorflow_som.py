"""
Generate SOMs from the tensorflow implementation of SOM.
"""
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


def test_som_transformation(sample):
    fcsdata = sample.data

    sample_size = 5000

    model = flowcat.models.FCSSom(
        (32, 32, -1), markers=sample.markers,
        batch_size=5001, tube=sample.tube,
        max_epochs=3)
    model.train([fcsdata], sample=sample_size)

    weights = model.weights
    model2 = flowcat.models.FCSSom(
        (32, 32, -1), init=("reference", weights),
        max_epochs=2, scaler=model.scaler,
        batch_size=50000)
    result = model2.transform(fcsdata)
    result2 = model2.transform(fcsdata)

    model3 = flowcat.models.FCSSom(
        (32, 32, -1), init=("reference", weights),
        max_epochs=2, scaler=model.scaler,
        batch_size=50000)
    result3 = model3.transform(fcsdata)

    print(result.data - result2.data)
    print(result.data - result3.data)


def train_native(sample, outname):
    fcsdata = sample.data

    sample_size = 5000

    model = flowcat.models.FCSSom(
        (32, 32, -1), markers=sample.markers,
        batch_size=5001, tube=sample.tube,
        max_epochs=3,
        tensorboard_dir=f"output/21-tensorboard/{outname}")
    weights = model.transform([fcsdata], sample=sample_size)

    flowcat.save_som(weights, f"output/21-tfsom/{outname}")


def compare_weights(weight_a, weight_b):
    print(weight_a.data - weight_b.data)


def train_loaded_check(trainsample, transsample):
    if flowcat.utils.URLPath("output/21-tfsom/temp_t1.csv").exists():
        print("Loading existing. Checking if existing model is used correctly.")
        refsom = flowcat.load_som("output/21-tfsom/temp", tube=1)
        loaded_model = flowcat.models.FCSSom((32, 32, -1), init=("reference", refsom), max_epochs=3, batch_size=5001)

        model = flowcat.models.FCSSom(
            (32, 32, -1), markers=trainsample.markers,
            batch_size=5001, tube=trainsample.tube,
            max_epochs=3,
        )
        model.train([trainsample.data])

        loaded_weights = loaded_model.transform(transsample.data)
        weights = model.transform(transsample.data)
        compare_weights(loaded_weights, weights)
    else:
        print("Creating new")
        model = flowcat.models.FCSSom(
            (32, 32, -1), markers=trainsample.markers,
            batch_size=5001, tube=trainsample.tube,
            max_epochs=3,
        )

        model.train([trainsample.data])
        flowcat.save_som(model.weights, f"output/21-tfsom/temp")

    weights = model.transform(transsample.data)
    print(weights)


if __name__ == "__main__":
    configure_print_logging()

    ds_missing = flowcat.CaseCollection.from_path("output/missing")
    ds_subsample = flowcat.CaseCollection.from_path("output/subsample")
    sample_missing = ds_missing.data[0].get_tube(1)
    sample_subsample = ds_subsample.data[0].get_tube(1)

    # train_native(sample_missing, "native-missing-test")
    # train_native(sample_subsample, "native-subsample")
    # test_som_transformation(sample_subsample)

    train_loaded_check(sample_subsample, sample_missing)
