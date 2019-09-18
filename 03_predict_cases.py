#!/usr/bin/env python3
import sys
import math
import os
import logging

import numpy as np
from sklearn import metrics
from tensorflow import keras
from argmagic import argmagic

from flowcat import utils, io_functions, som_dataset, mappings, classification_utils


LOGGER = logging.getLogger(__name__)


def setup_logging_from_args(args):
    logpath = args.logpath
    if logpath:
        logpath = utils.URLPath(logpath)
        logpath.local.parent.mkdir(parents=True, exist_ok=True)
    return setup_logging(logpath, printlevel=args_loglevel(args.verbose))


def setup_logging(filelog=None, filelevel=logging.DEBUG, printlevel=logging.WARNING):
    """Setup logging to both visible output and file output.
    Args:
        filelog: Logging file. Will not log to file if None
        filelevel: Logging level inside file.
        printlevel: Logging level for visible output.
    """
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=printlevel),
    ]
    if filelog is not None:
        handlers.append(
            utils.create_handler(logging.FileHandler(str(filelog)), level=filelevel)
        )

    utils.add_logger("flowcat", handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def args_loglevel(vlevel):
    """Get logging level from number of verbosity chars."""
    if not vlevel:
        return logging.WARNING
    if vlevel == 1:
        return logging.INFO
    return logging.DEBUG


def main(
        data: utils.URLPath,
        meta: utils.URLPath,
        labels: utils.URLPath,
        model: utils.URLPath,
):
    dataset = io_functions.load_case_collection(data, meta)
    test_labels = io_functions.load_json(labels)
    dataset = dataset.filter(labels=test_labels)

    model_config = io_functions.load_json(model / "config.json")
    tubes = list(model_config["tubes"].keys())
    pad_width = model_config["pad_width"]
    groups = model_config["groups"]

    group_mapping = model_config["mapping"]
    mapping = group_mapping["map"]
    if mapping:
        dataset = dataset.map_groups(mapping)

    def bloodcall(case):
        return case.used_material == mappings.Material.PERIPHERAL_BLOOD
    bloodset = dataset.filter(custom_callback=bloodcall)

    print(np.mean([c.infiltration for c in bloodset]))

    def kmcall(case):
        return case.used_material == mappings.Material.BONE_MARROW
    kmset = dataset.filter(custom_callback=kmcall)
    print(np.mean([c.infiltration for c in kmset]))
    return

    binarizer = io_functions.load_joblib(model / "binarizer.joblib")

    def getter_fun(data, tube):
        return data.get_tube(tube, kind="som").get_data().data

    bseq = som_dataset.SOMSequence(
        bloodset, binarizer,
        tube=tubes,
        get_array_fun=getter_fun,
        batch_size=128,
        pad_width=pad_width)
    kseq = som_dataset.SOMSequence(
        kmset, binarizer,
        tube=tubes,
        get_array_fun=getter_fun,
        batch_size=128,
        pad_width=pad_width)
    model = keras.models.load_model(model / "model.h5", custom_objects={"loss": classification_utils.WeightedCategoricalCrossentropy()})

    preds = []
    for pred in model.predict_generator(bseq):
        preds.append(pred)
    pred_arr = np.array(preds)
    pred_labels = binarizer.inverse_transform(pred_arr)
    true_labels = bseq.true_labels

    confusion = metrics.confusion_matrix(true_labels, pred_labels, labels=groups)
    acc = metrics.accuracy_score(true_labels, pred_labels)
    print(groups, acc)
    print(confusion)

    preds = []
    for pred in model.predict_generator(kseq):
        preds.append(pred)
    pred_arr = np.array(preds)
    pred_labels = binarizer.inverse_transform(pred_arr)
    true_labels = kseq.true_labels

    confusion = metrics.confusion_matrix(true_labels, pred_labels, labels=groups)
    acc = metrics.accuracy_score(true_labels, pred_labels)
    print(groups, acc)
    print(confusion)

if __name__ == "__main__":
    argmagic(main)
