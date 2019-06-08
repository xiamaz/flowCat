"""
Generate SOMs from the tensorflow implementation of SOM.
"""
import logging
import collections
import time
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
    print(model.model.output_weights)
    model.save("output/test_model_save/som_transformation")

    weights = model.weights
    model2 = flowcat.models.FCSSom(
        (32, 32, -1), init=("reference", weights),
        max_epochs=2, scaler=model.scaler,
        tensorboard_dir="output/21-tensorboard/test_som_transformation",
        batch_size=100000)
    result = model2.transform(fcsdata, label="t1")
    result2 = model2.transform(fcsdata, label="t2")
    print(result.data - result2.data)
    print("------")

    # model3 = flowcat.models.FCSSom(
    #     (32, 32, -1), init=("reference", weights),
    #     max_epochs=2, scaler=model.scaler,
    #     batch_size=50000)
    # result3 = model3.transform(fcsdata)

    # models = flowcat.models.FCSSom.load("output/test_model_save/som_transformation", batch_size=50000, max_epochs=2)
    # results = models.transform(fcsdata)

    # print(result.data - result2.data)
    # print(result.data - result3.data)
    # print(result.data - results.data)


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


def time_generator_logger(generator):
    circ_buffer = collections.deque(maxlen=20)
    time_a = time.time()
    for res in generator:
        time_b = time.time()
        time_d = time_b - time_a
        circ_buffer.append(time_d)
        time_rolling = np.mean(circ_buffer)
        print(f"Training time: {time_d}s Rolling avg: {time_rolling}s")
        time_a = time_b
        yield time_d, time_rolling, res


def test_batch_transform_speed(tubedata, fitmap: bool):
    """Test the speed for transforming a list of samples.

    Take the time we need to get a whole sample.
    """

    model = flowcat.models.FCSSom.load(f"output/test_model_save/native-subsample", batch_size=50000)
    res = []
    if fitmap:
        for _, mean, somdata in time_generator_logger(model.transform_fitmap(tubedata)):
            res.append(somdata)
    else:
        for _, mean, somdata in time_generator_logger(model.transform_multiple(tubedata)):
            res.append(somdata)

    return mean, res


def compare_som(som_a, som_b):
    print(som_a.data - som_b.data)


def compare_fitmap(dataset):
    view = dataset.filter(num=3, groups=["CLL", "normal"])
    print(view)
    print(dataset.filter_failed())
    tube_1 = view.get_tube(1)

    mean_fitmap, res_fitmap = test_batch_transform_speed(tube_1, fitmap=False)
    mean_direct, res_direct = test_batch_transform_speed(tube_1, fitmap=True)
    print(f"Fitmap rolling mean: {mean_fitmap}s Direct mean: {mean_direct}")
    for som_a, som_b in zip(res_fitmap, res_direct):
        compare_som(som_a, som_b)


def test_refactor(dataset):
    view = dataset.filter(num=3, groups=["CLL", "normal"])
    sample = view.data[0]

    model = flowcat.som.casesom.CaseSingleSom(
        dims=(32, 32, -1),
        markers=sample.get_tube(1).markers,
        tube=1,
        materials=flowcat.ALLOWED_MATERIALS,
        max_epochs=3,
        batch_size=50000)
    model.train([sample])

    for res in time_generator_logger(model.transform_generator(view)):
        print(res)

    print(model.weights)


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

    # compare_fitmap(ds_subsample)
    test_refactor(ds_subsample)
