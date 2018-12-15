#!/usr/bin/env python3
import sys
import math
import os
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn import metrics

import keras
from keras import models, optimizers
from keras.utils import plot_model

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

from flowcat import utils, configuration, loaders
from flowcat.dataset import combined_dataset
from flowcat.mappings import NAME_MAP, GROUP_MAPS
from flowcat.visual import plotting

from flowcat.models import weighted_crossentropy
from flowcat.models import fcs_cnn, histo_nn, som_cnn


COLS = "grcmyk"
LOGGER = logging.getLogger(__name__)


def inverse_binarize(y, classes):
    classes = np.asarray(classes)
    if isinstance(y, pd.DataFrame):
        y = y.values
    if len(classes) > 1:
        return classes.take(y.argmax(axis=1), mode="raise")
    raise RuntimeError("Cannot invert less than 2 classes.")


def get_weights_by_name(name, groups):
    if name == "weighted":
        # Group weights are a dict mapping tuples to tuples. Weights are for
        # false classifications in the given direction.
        # (a, b) --> (a>b, b>a)
        group_weights = {
            ("normal", None): (5.0, 10.0),
            ("MZL", "LPL"): (2, 2),
            ("MCL", "PL"): (2, 2),
            ("FL", "LPL"): (3, 5),
            ("FL", "MZL"): (3, 5),
        }
        weights = create_weight_matrix(group_weights, groups, base_weight=5)
    elif name == "simpleweights":
        # simpler group weights
        group_weights = {
            ("normal", None): (1.0, 20.0),
        }
        weights = create_weight_matrix(group_weights, groups, base_weight=1)
    elif name == "normalweights":
        group_weights = {
            ("normal", None): (10.0, 10.0),
        }
        weights = create_weight_matrix(group_weights, groups, base_weight=1)
    else:
        weights = None
    return weights


def plot_transformed(path, tf1, tf2, y):
    """Plot transformed data with colors for labels."""

    path.mkdir(parents=True, exist_ok=True)
    figpath = path / "decompo"

    fig = Figure(figsize=(10, 5))

    axes = fig.add_subplot(121)
    for i, group in enumerate(y.unique()):
        sel_data = tf1[y == group]
        axes.scatter(sel_data[:, 0], sel_data[:, 1], label=group, c=COLS[i])
    axes.legend()

    axes = fig.add_subplot(122)
    for i, group in enumerate(y.unique()):
        sel_data = tf2[y == group]
        axes.scatter(sel_data[:, 0], sel_data[:, 1], label=group, c=COLS[i])
    axes.legend()

    FigureCanvas(fig)
    fig.savefig(figpath)


def plot_train_history(path, data):
    """Plot the training history of the given data."""

    fig = Figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)

    # Traning dataset loss and accuracy metrics
    ax.plot(
        range(len(data["loss"])), data["loss"],
        c="blue", linestyle="--", label="Loss")
    if "val_loss" in data:
        ax.plot(
            range(len(data["loss"])), data["val_loss"],
            c="red", linestyle="--", label="Validation Loss")

    # Testing dataset loss and accuracy metrics
    ax.plot(
        range(len(data["acc"])), data["acc"],
        c="blue", linestyle="-", label="Accuracy")
    if "val_acc" in data:
        ax.plot(
            range(len(data["val_acc"])),
            data["val_acc"],
            c="red", linestyle="-", label="Validation Accuracy")

    ax.set_xlabel("No. Epoch")
    ax.set_ylabel("Loss value / Acc")

    ax.legend()

    FigureCanvas(fig)

    fig.savefig(path)


def save_model(model, path, name):
    """Save the given model to the given path."""
    # plot a model diagram
    plotpath = path / f"modelplot_{name}.png"
    plotpath.put(lambda p: keras.utils.plot_model(model, p, show_shapes=True))

    modelpath = path / f"model_{name}.h5"
    modelpath.put(lambda p: model.save(p))


def save_predictions(df, path, name):
    """Save the given predictions into a specified csv file."""
    predpath = path / f"predictions_{name}.csv"
    predpath.put(lambda p: df.to_csv(str(p)))


def save_history(history, path, name):
    """Save train history."""
    histpath = path / f"trainhistory_{name}"
    histpath.put(lambda p: plot_train_history(p, history.history))
    utils.save_pickle(history.history, path / f"history_{name}.p")


def load_weights(weights):
    """Load weights using named labels or load matrices from paths."""
    # TODO implement real functionality
    if weights is not None:
        raise RuntimeError("Weights have not been implemented yet.")
    return weights


def save_weights(weights, path, name):
    """Save weighted matrix."""
    if weights is not None:
        weights.to_csv(path / f"weights_{name}.csv")
        plotting.plot_confusion_matrix(
            weights.values, weights.columns, normalize=False, cmap=cm.get_cmap("Reds"),
            title="Weight Matrix",
            filename=path / f"weightsplot_{name}", dendroname=None)


def run_model(
        model, trainseq, testseq,
        train_epochs=200, epsilon=1e-8, initial_rate=1e-3, drop=0.5, epochs_drop=100, num_workers=1,
        validation=False, weights=None, pregenerate=False):
    """Run and predict using the given model."""

    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights.values)

    def top2_acc(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    def create_stepped(initial_rate=1e-3, drop=0.5, epochs_drop=100):

        def scheduler(epoch):
            lrate = initial_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        return keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(
        loss=lossfun,
        # optimizer="adam",
        optimizer=optimizers.Adam(lr=0.0, decay=0.0, epsilon=epsilon),  # lr and decay set by callback
        metrics=[
            "acc",
            # top2_acc,
        ]
    )

    if pregenerate:
        trainseq.generate_batches(num_workers)
        testseq.generate_batches(num_workers)
        LOGGER.info("Pregenerating batches in sequences. Worker number will be ignored in the fit generator")
        num_workers = 0

    history = model.fit_generator(
        trainseq, epochs=train_epochs,
        callbacks=[
            # keras.callbacks.EarlyStopping(min_delta=0.01, patience=20, mode="min"),
            create_stepped(initial_rate, drop, epochs_drop),
        ],
        validation_data=testseq if validation else None,
        # class_weight={
        #     0: 1.0,  # CM
        #     1: 2.0,  # MCL
        #     2: 2.0,  # PL
        #     3: 2.0,  # LP
        #     4: 2.0,  # MZL
        #     5: 50.0,  # FL
        #     6: 50.0,  # HCL
        #     7: 1.0,  # normal
        # }
        workers=num_workers,
        use_multiprocessing=False,
        shuffle=True,
    )
    pred_mat = model.predict_generator(testseq, workers=num_workers, use_multiprocessing=True)

    pred_df = pd.DataFrame(
        pred_mat, columns=testseq.groups, index=testseq.labels)
    pred_df["correct"] = testseq.ylabels
    return model, pred_df, history


def merge_predictions(prediction, group_map, groups):
    new_predictions = prediction.copy()
    for group in groups:
        if group in group_map.values() and group not in new_predictions.columns:
            assoc_groups = [k for k, v in group_map.items() if v == group and k in new_predictions.columns]
            if not assoc_groups:
                continue
            new_predictions[group] = new_predictions.loc[:, assoc_groups].sum(axis=1)
            new_predictions.drop(assoc_groups, inplace=True, axis=1)
    new_predictions = new_predictions.loc[:, groups]
    mapfun = np.vectorize(lambda l: group_map.get(l, l))
    new_predictions["correct"] = mapfun(prediction["correct"])
    return new_predictions


def create_metrics_from_pred(pred_df, mapping=None):
    """Create metrics from pred df."""
    groups = [c for c in pred_df.columns if c != "correct"]

    if mapping is not None:
        merged_groups = []
        for group in groups:
            mapped = mapping.get(group, group)
            if mapped not in merged_groups:
                merged_groups.append(mapped)
        groups = merged_groups

        pred_df = merge_predictions(pred_df, mapping, groups)

    pred = inverse_binarize(pred_df.loc[:, groups].values, classes=groups)
    corr = pred_df["correct"].values

    stats = {}
    weighted_f1 = metrics.f1_score(corr, pred, average="weighted")
    unweighted_f1 = metrics.f1_score(corr, pred, average="macro")
    mcc = metrics.matthews_corrcoef(corr, pred)
    stats["weighted_f1"] = weighted_f1
    stats["unweighted_f1"] = unweighted_f1
    stats["mcc"] = mcc
    LOGGER.info("weighted F1: %f", weighted_f1)
    LOGGER.info("unweighted F1: %f", unweighted_f1)
    LOGGER.info("MCC: %f", mcc)

    confusion = metrics.confusion_matrix(corr, pred, groups,)
    confusion = pd.DataFrame(confusion, columns=groups, index=groups)
    LOGGER.info("Confusion matrix")
    LOGGER.info(confusion)
    return confusion, stats


def create_weight_matrix(group_map, groups, base_weight=5):
    """Generate weight matrix from given group mapping."""
    # expand mapping to all other groups if None is given
    expanded_groups = {}
    for (group_a, group_b), v in group_map.items():
        if group_a is None:
            for g in groups:
                if g != group_b:
                    expanded_groups[(g, group_b)] = v
        elif group_b is None:
            for g in groups:
                if g != group_a:
                    expanded_groups[(group_a, g)] = v
        else:
            expanded_groups[(group_a, group_b)] = v

    weights = base_weight * np.ones((len(groups), len(groups)))
    for i in range(len(groups)):
        weights[i, i] = 1
    for (group_a, group_b), (ab_err, ba_err) in expanded_groups.items():
        weights[groups.index(group_a), groups.index(group_b)] = ab_err
        weights[groups.index(group_b), groups.index(group_a)] = ba_err
    weights = pd.DataFrame(weights, columns=groups, index=groups)
    return weights


def create_output_spec(modelname, dataoptions):
    if modelname == "histogram":
        partial_spec = [
            (loaders.CountLoader.create_inferred, {"tube": 1}),
            (loaders.CountLoader.create_inferred, {"tube": 2}),
        ]
    elif modelname == "som":
        partial_spec = [
            (loaders.Map2DLoader.create_inferred, {"tube": 1}),
            (loaders.Map2DLoader.create_inferred, {"tube": 2}),
        ]
    elif modelname == "etefcs":
        partial_spec = [
            (loaders.FCSLoader.create_inferred, {"tubes": [1, 2]}),
        ]
    else:
        raise RuntimeError(f"Unknown model {modelname}")

    # create partial loader functions
    output_spec = [loaders.loader_builder(f, **{**v, **dataoptions[f.__self__.__name__]}) for f, v in partial_spec]

    return output_spec


def generate_model_inputs(
        train_data, test_data, config, path=None):
    """Create model and associated inputs.
    Args:
        train_data, test_data: Dataset used to infer sizes.
        name: String name of model.
        config: Configuration object.
        path: Optional path to load previous model from.
    Returns:
        Tuple of model and partial dataset constructor functions.
    """
    mtype = config["type"]
    loader_options = config["loader"]
    traindata_args = config["train_args"]
    testdata_args = config["test_args"]
    output_spec = create_output_spec(mtype, loader_options)
    train = loaders.DatasetSequence(train_data, output_spec, **traindata_args)
    test = loaders.DatasetSequence(test_data, output_spec, **testdata_args)
    if path is not None:
        model = keras.models.load_model(path, compile=False)
    else:
        constructors = {
            "histogram": histo_nn.create_model_histo,
            "som": som_cnn.create_model_cnn,
            "etefcs": fcs_cnn.create_model_fcs,
        }
        model = constructors[mtype](*train.shape)

    return model, train, test


def create_stats(outpath, dataset, pred_df, confusion_sizes):
    LOGGER.info("=== Statistics results ===")
    for gname, groupstat in GROUP_MAPS.items():
        # skip if our cohorts are larger
        if len(dataset.mapping["groups"]) < len(groupstat["groups"]):
            continue

        LOGGER.info(f"-- {len(groupstat['groups'])} --")
        conf, stats = create_metrics_from_pred(pred_df, mapping=groupstat["map"])

        # choose whether to represent merged classes larger in confusion matrix
        sizes = groupstat["sizes"] if confusion_sizes else None

        title = f"Confusion matrix (weighted f1 {stats['weighted_f1']:.2f} unweighted f1 {stats['unweighted_f1']:.2f})"
        plotpath = outpath / f"confusion_{gname}.png"
        plotpath.put(
            lambda p: plotting.plot_confusion_matrix(
                conf.values, groupstat["groups"], normalize=True,
                title=title, filename=p, dendroname=None, sizes=sizes))
        utils.save_csv(conf, outpath / f"confusion_{gname}.csv")
        utils.save_json(stats, outpath / "stats_{gname}.json")


def setup_logging(filelog=None, filelevel=logging.DEBUG, printlevel=logging.WARNING):
    """Setup logging to both visible output and file output.
    Args:
        filelog: Logging file. Will not log to file if None
        filelevel: Logging level inside file.
        printlevel: Logging level for visible output.
    """
    handlers = [
        create_handler(logging.StreamHandler(), level=printlevel),
    ]
    if filelog is not None:
        handlers.append(
            create_handler(logging.FileHandler(str(filelog)), level=filelevel)
        )

    add_logger("flowcat", handlers, level=logging.DEBUG)
    add_logger(LOGGER, handlers, level=logging.DEBUG)


class SOMClassifierConfig(configuration.ClassificationConfig):
    """Generate configuration for SOM classification"""

    @classmethod
    def generate_config(cls):

        name = "test"
        dataset_name = "selected1_toroid_s32"
        is_toroid = "toroid" in dataset_name
        train_size = 64
        test_size = 32

        data = {
            "name": name,
            "dataset": {
                "names": {
                    "FCS": "fixedCLL-9F",
                    "SOM": dataset_name,
                },
                "filters": {
                    "counts": 10000,
                    "groups": ["CLL", "MBL", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"],
                    "num": None,
                },
                "mapping": "8class",
            },
            "split": {
                "train_num": 0.9,
                "train_labels": None,
                "test_labels": None,
            },
            "model": {
                "type": "som",
                "loader": {
                    "Map2DLoader": {
                        "sel_count": None,  # use count in SOM map as just another channel
                        "pad_width": 1 if is_toroid else 0,
                    },
                },
                "train_args": {
                    "batch_size": train_size,
                    "draw_method": "balanced",
                    "epoch_size": 16000,
                    "sample_weights": False,
                },
                "test_args": {
                    "batch_size": test_size,
                    "draw_method": "sequential",
                    "epoch_size": None,
                },
            },
            "run": {
                "train_epochs": 100,
                "validation": False,
                "pregenerate": True,
            }
        }
        return cls(data)


def args_loglevel(vlevel):
    """Get logging level from number of verbosity chars."""
    if not vlevel:
        return logging.WARNING
    if vlevel == 1:
        return logging.INFO
    return logging.DEBUG


def run(args):
    config = args.modelconfig
    pathconfig = args.pathconfig

    if args.name:
        config.data["name"] = args.name
    classification_path = pathconfig("output", "classification")
    outpath = utils.URLPath(f"{classification_path}/{config('name')}")

    # Create logfiles
    logpath = outpath / f"classification.log"
    logpath.local.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(logpath, printlevel=args_loglevel(args.verbose))

    # save configuration
    config.to_file(outpath / "config.toml")

    dataset = combined_dataset.CombinedDataset.from_config(pathconfig, config)

    # split into train and test set
    LOGGER.info("Splitting dataset")
    train, test = combined_dataset.split_dataset(dataset, **config("split"), seed=args.seed)
    utils.save_json(train.labels, outpath / "train_labels.json")
    utils.save_json(test.labels, outpath / "test_labels.json")

    LOGGER.info("Getting models")
    model, trainseq, testseq = generate_model_inputs(train, test, config("model"))

    LOGGER.info("Running model")
    weights = load_weights(config("run", "weights"))
    model, pred_df, history = run_model(
        model, trainseq, testseq, **{**config("run"), "weights": weights})
    save_model(model, outpath, name="0")
    save_history(history, outpath, name="0")
    save_weights(weights, outpath, name="0")
    save_predictions(pred_df, outpath, name="0")

    LOGGER.info("Creating statistics")
    create_stats(
        outpath=outpath, dataset=dataset, pred_df=pred_df, **config("stat"))


def config(args):
    setup_logging(filelog=None, printlevel=logging.ERROR)
    config = getattr(args, args.type)

    if args.output is None:
        print("HINT: Redirect stderr to get config output suitable for piping.", file=sys.stderr)
        print(config)
    else:
        print(f"Writing configuration to {args.output}")
        config.to_file(args.output)


def main():
    parser = argparse.ArgumentParser(description="Classify samples")
    parser.add_argument(
        "--pathconfig",
        help="Config file containing paths.",
        type=lambda p: configuration.PathConfig.from_file(p) if p else configuration.PathConfig({}),
        default="paths.toml")
    parser.add_argument(
        "--modelconfig",
        help="Path to configuration file.",
        type=lambda p: SOMClassifierConfig.from_file(p) if p else SOMClassifierConfig.generate_config(),
        default="")
    subparsers = parser.add_subparsers()

    # Save configuration files in specified in the create configuration file
    cparser = subparsers.add_parser(
        "config",
        help="Output the current configuration. Hint: Get rid of import messages by eg piping stderr to /dev/null")
    cparser.add_argument(
        "-o", "--output",
        help="Write configuration to output file.",
        type=utils.URLPath)
    cparser.add_argument(
        "--type",
        help="Output pathconfig instead of model configuration.",
        choices=["pathconfig", "modelconfig"],
        default="modelconfig",
    )
    cparser.set_defaults(fun=config)

    # Run the classification process
    rparser = subparsers.add_parser("run", help="Run classification")
    rparser.add_argument(
        "--seed",
        help="Seed for random number generator",
        type=int)
    rparser.add_argument(
        "--model",
        help="Use an existing model",
        type=utils.URLPath) # TODO
    rparser.add_argument(
        "--name",
        help="Set an alternative output name.")
    rparser.add_argument(
        "-v", "--verbose",
        help="Control verbosity. -v is info, -vv is debug",
        action="count")
    rparser.set_defaults(fun=run)

    args = parser.parse_args()

    if not hasattr(args, "fun") or args.fun is None:
        parser.print_help()
    else:
        args.fun(args)


if __name__ == "__main__":
    main()
