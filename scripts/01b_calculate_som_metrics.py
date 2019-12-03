#!/usr/bin/env python3
"""Calculate metrics for comparing different SOM options on the dataset."""
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from argmagic import argmagic

from flowcat import utils, io_functions
from flowcat.dataset.som import SOM


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


def sample_quantization_error(fcsdata, somdata, model, session):
    fcsdata = fcsdata.align(somdata.markers).data
    fcsdata = StandardScaler().fit_transform(fcsdata)
    somdata = somdata.data.reshape((-1, len(somdata.markers)))
    error = session.run(model, feed_dict={"fcs:0": fcsdata, "som:0": somdata})
    return float(error)


def load_npy(filepath, markers):
    data = np.load(str(filepath))
    return SOM(data, markers)


def load_csv(filepath, _):
    data = io_functions.load_csv(filepath)
    return SOM(data.values, list(data.columns))


def get_som_data(case_id, tube, data_path, som_markers):
    filename = f"{case_id}_t{tube}"
    filepath = data_path / filename

    for suffix, loadfun in ((".csv", load_csv), (".npy", load_npy)):
        fpath = filepath + suffix
        if fpath.exists():
            data = loadfun(fpath, som_markers)
            break

    return data


def main(
        fcsdata: utils.URLPath,
        fcsmeta: utils.URLPath,
        somdata: utils.URLPath,
        output: utils.URLPath,
):

    fcs_dataset = io_functions.load_case_collection(fcsdata, fcsmeta)
    try:
        som_config = io_functions.load_json(somdata + "_config.json")
    except FileNotFoundError:
        som_config = None

    if som_config is None:
        selected_markers = fcs_dataset.selected_markers
    else:
        selected_markers = {t: d["channels"] for t, d in som_config.items()}

    tubes = ("1", "2", "3")

    model = quantization_error_model()
    sess = tf.Session()
    results = []
    for fcscase in fcs_dataset:
        print(fcscase)
        for tube in tubes:
            fcssample = fcscase.get_tube(tube, kind="fcs").get_data()
            somsample = get_som_data(fcscase.id, tube, somdata, selected_markers[tube])
            error = sample_quantization_error(fcssample, somsample, model, sess)
            results.append((fcscase.id, tube, error))

    stats = {}
    stats["mean"] = {
        t: sum(r[-1] for r in results if r[1] == t) / len(results)
        for t in tubes
    }
    stats["variance"] = {
        t: sum(np.power(r[-1] - stats["mean"][t], 2) for r in results if r[1] == t) / len(results)
        for t in tubes
    }
    print("Mean quantization error", stats)

    io_functions.save_json(results, output / "quantization_error.json")
    io_functions.save_json(stats, output / "quantization_error_mean.json")


argmagic(main)
