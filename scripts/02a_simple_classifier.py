#!/usr/bin/env python3
"""
Train models on Munich data and attempt to classify Bonn data.
"""
import json
import numpy as np
import pandas as pd
from sklearn import metrics, neighbors, ensemble, naive_bayes
from sklearn.preprocessing import LabelBinarizer
from argmagic import argmagic

from flowcat import utils, io_functions, mappings, classification_utils
from flowcat.som_dataset import SOMDataset, SOMSequence
from flowcat.plots import confusion as plot_confusion, history as plot_history


MODELS = {
    "RandomForest": ensemble.RandomForestClassifier,
    "NaiveBayes": naive_bayes.GaussianNB,
    "kNN": neighbors.KNeighborsClassifier,
}


def get_model(channel_config, groups, model_name="kNN", **kwargs):
    # inputs = tuple([*d["dims"][:-1], len(d["channels"])] for d in channel_config.values())
    # output = len(groups)

    model = MODELS[model_name](**kwargs)

    binarizer = LabelBinarizer()
    binarizer.fit(groups)
    return binarizer, model


def map_labels(labels, mapping):
    """Map labels to new labels defined by mapping."""
    return [mapping.get(l, l) for l in labels]


def generate_confusion(true_labels, pred_labels, groups, output):
    """Calculate confusion matrix metrics and also create plots."""
    confusion = metrics.confusion_matrix(true_labels, pred_labels, labels=groups)
    confusion = pd.DataFrame(confusion, index=groups, columns=groups)
    print(confusion)
    io_functions.save_csv(confusion, output / "validation_confusion.csv")

    plot_confusion.plot_confusion_matrix(
        confusion, normalize=False, filename=output / "confusion_abs.png",
        dendroname="dendro.png"
    )
    plot_confusion.plot_confusion_matrix(
        confusion, normalize=True, filename=output / "confusion_norm.png",
        dendroname=None
    )
    return confusion


def generate_metrics(true_labels, pred_labels, groups, output):
    """Generate numeric metrics."""
    metrics_results = {
        "balanced": metrics.balanced_accuracy_score(true_labels, pred_labels),
        "f1_micro": metrics.f1_score(true_labels, pred_labels, average="micro"),
        "f1_macro": metrics.f1_score(true_labels, pred_labels, average="macro"),
        "mcc": metrics.matthews_corrcoef(true_labels, pred_labels),
    }
    print(metrics_results)
    io_functions.save_json(metrics_results, output / "validation_metrics.json")
    return metrics_results


def generate_all_metrics(true_labels, pred_labels, mapping, output):
    output.mkdir()

    groups = mapping["groups"]
    map_dict = mapping["map"]
    if map_dict:
        true_labels = map_labels(true_labels, map_dict)
        pred_labels = map_labels(pred_labels, map_dict)

    confusion = generate_confusion(true_labels, pred_labels, groups, output)
    metrics = generate_metrics(true_labels, pred_labels, groups, output)

    return confusion, metrics


def plot_training_history(history, output):
    history_data = {
        "accuracy": history.history["categorical_accuracy"],
        "val_accuracy": history.history["val_categorical_accuracy"],
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
    }
    acc_plot = plot_history.plot_history(history_data, title="Training accuracy")
    acc_plot.savefig(str(output), dpi=300)


def som2d_to_1d(som):
    w, h, c = som.shape
    return som.reshape((w * h * c))


def concatenate_batch(batch):
    result = np.concatenate([
        np.stack([som2d_to_1d(b) for b in t])
        for t in batch
    ], axis=-1)
    return result


def sequence_to_array(sequence):
    batches = [sequence[i] for i in range(len(sequence))]
    xdata = np.concatenate(
        [
            concatenate_batch(s) for s, _ in batches
        ]
    )
    print(xdata.shape)
    ydata = np.concatenate([s[1] for s in batches])
    print(ydata.shape)
    return xdata, ydata


def main(data: utils.URLPath, output: utils.URLPath, model_name: str, modelargs: json.loads, epochs: int = 30):
    """
    Args:
        data: Path to som dataset
        output: Output path
    """
    tubes = ("1", "2", "3")
    pad_width = 0

    group_mapping = mappings.GROUP_MAPS["8class"]
    mapping = group_mapping["map"]
    groups = group_mapping["groups"]
    # mapping = None
    # groups = mappings.GROUPS

    dataset = SOMDataset.from_path(data)
    if mapping:
        dataset = dataset.map_groups(mapping)

    dataset = dataset.filter(groups=groups)

    dataset_groups = {d.group for d in dataset}

    if set(groups) != dataset_groups:
        raise RuntimeError(f"Group mismatch: {groups}, but got {dataset_groups}")

    train, validate = dataset.create_split(0.9, stratify=True)

    # train = train.balance(20)
    train = train.balance_per_group({
        "CM": 6000,
        # "CLL": 4000,
        # "MBL": 2000,
        "MCL": 1000,
        "PL": 1000,
        "LPL": 1000,
        "MZL": 1000,
        "FL": 1000,
        "HCL": 1000,
        "normal": 6000,
    })

    io_functions.save_json(train.labels, output / "ids_train.json")
    io_functions.save_json(validate.labels, output / "ids_validate.json")

    som_config = io_functions.load_json(data + "_config.json")
    selected_tubes = {tube: som_config[tube] for tube in tubes}

    config = {
        "tubes": selected_tubes,
        "groups": groups,
        "pad_width": pad_width,
        "mapping": group_mapping,
        "cost_matrix": None,
    }
    io_functions.save_json(config, output / "config.json")

    for tube in tubes:
        x, y, z = selected_tubes[tube]["dims"]
        selected_tubes[tube]["dims"] = (x + 2 * pad_width, y + 2 * pad_width, z)

    # binarizer, model = get_model(selected_tubes, groups=groups, n_neighbors=1)
    binarizer, model = get_model(selected_tubes, groups=groups, model_name="RandomForest", **modelargs)

    def getter_fun(sample, tube):
        return sample.get_tube(tube)

    trainseq = SOMSequence(
        train,
        binarizer,
        tube=tubes,
        get_array_fun=getter_fun,
        batch_size=32,
        pad_width=pad_width)
    validseq = SOMSequence(
        validate,
        binarizer,
        tube=tubes,
        get_array_fun=getter_fun,
        batch_size=128,
        pad_width=pad_width)

    xdata, ydata = sequence_to_array(trainseq)

    model.fit(xdata, ydata)

    xtest, ytest = sequence_to_array(validseq)
    pred_arr = model.predict(xtest)

    io_functions.save_joblib(binarizer, output / "binarizer.joblib")

    pred_labels = binarizer.inverse_transform(pred_arr)
    true_labels = validseq.true_labels

    generate_all_metrics(
        true_labels, pred_labels, {"groups": groups, "map": {}}, output / "unmapped")
    for map_name, mapping in mappings.GROUP_MAPS.items():
        output_path = output / map_name
        # skip if more groups in map
        print(f"--- MAPPING: {map_name} ---")
        if len(mapping["groups"]) > len(groups):
            continue
        generate_all_metrics(true_labels, pred_labels, mapping, output_path)


if __name__ == "__main__":
    argmagic(main)
