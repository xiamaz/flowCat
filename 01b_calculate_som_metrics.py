"""Calculate metrics for comparing different SOM options on the dataset."""
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from argmagic import argmagic

from flowcat import utils, io_functions


def quantization_error_model():
    mapdata = tf.placeholder(tf.float32, shape=(None, None), name="som")
    fcsdata = tf.placeholder(tf.float32, shape=(None, None), name="fcs")
    squared_diffs = tf.pow(tf.subtract(
        tf.expand_dims(mapdata, axis=0),
        tf.expand_dims(fcsdata, axis=1)), 2)
    diffs = tf.reduce_sum(squared_diffs, 2)
    euc_distance = tf.sqrt(tf.reduce_min(diffs, axis=1))
    qe = tf.reduce_mean(euc_distance)
    return qe


def generate_matched(*datasets):
    for case in datasets[0]:
        matched = [d.get_label(case.id) for d in datasets[1:]]
        yield (case, *matched)


def sample_quantization_error(fcssample, somsample, model, session):
    somdata = somsample.get_data()
    fcsdata = fcssample.get_data().align(somdata.markers).data
    fcsdata = MinMaxScaler().fit_transform(fcsdata)
    somdata = somdata.data.reshape((-1, somsample.dims[-1]))
    error = session.run(model, feed_dict={"fcs:0": fcsdata, "som:0": somdata})
    return error


def main(
        fcsdata: utils.URLPath,
        fcsmeta: utils.URLPath,
        somdata: utils.URLPath,
        sommeta: utils.URLPath,
        output: utils.URLPath,
):

    fcs_dataset = io_functions.load_case_collection(fcsdata, fcsmeta)
    som_dataset = io_functions.load_case_collection(somdata, sommeta)
    tubes = ("1", "2", "3")

    model = quantization_error_model()
    sess = tf.Session()
    results = []
    for fcscase, somcase in generate_matched(fcs_dataset, som_dataset):
        print(fcscase)
        for tube in tubes:
            fcssample = fcscase.get_tube(tube, kind="fcs")
            somsample = somcase.get_tube(tube, kind="som")
            error = sample_quantization_error(fcssample, somsample, model, sess)
            results.append((fcscase.id, tube, error))

    io_functions.save_json(results, output / "quantization_error.json")


argmagic(main)
