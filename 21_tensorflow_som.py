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
    model.train([fcsdata], sample=sample_size)

    flowcat.save_som(model.weights, f"output/21-tfsom/{outname}")


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


def test_model_save(trainsample, name):
    model = flowcat.models.FCSSom(
        (32, 32, -1), markers=trainsample.markers,
        batch_size=5000, tube=trainsample.tube,
        max_epochs=3,
    )
    model.train([trainsample.data])
    model.save(f"output/test_model_save/{name}")
    return model.transform(trainsample.data)


def test_model_load(trainsample, name):

    model = flowcat.models.FCSSom.load(f"output/test_model_save/{name}")
    weights = model.transform(trainsample.data)
    return weights


def test_batch_transform_speed(trainsamples: flowcat.CaseCollection):
    """Test the speed for transforming a list of samples.

    Take the time we need to get a whole sample.
    """
    print(trainsamples)
    view = trainsamples.filter()
    print(view)
    print(trainsamples.filter_failed())
    tube_1 = view.get_tube(1)
    print(tube_1)


if __name__ == "__main__":
    configure_print_logging()

    ds_missing = flowcat.CaseCollection.from_path("output/missing")
    ds_subsample = flowcat.CaseCollection.from_path("output/subsample")
    sample_missing = ds_missing.data[0].get_tube(1)
    sample_subsample = ds_subsample.data[0].get_tube(1)

    # train_native(sample_missing, "native-missing-test")
    # train_native(sample_subsample, "native-subsample")
    # test_som_transformation(sample_subsample)

    # weights = test_model_save(sample_subsample, "native-subsample")
    # loaded_weights = test_model_load(sample_subsample, "native-subsample")
    # print(weights.data - loaded_weights.data)

    # train_loaded_check(sample_subsample, sample_missing)

    test_batch_transform_speed(ds_subsample)
