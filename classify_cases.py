import math
import json
import os
import pickle
import pathlib
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn import manifold, model_selection, preprocessing
from sklearn import naive_bayes
from sklearn import metrics

import keras
from keras import layers, models, regularizers, optimizers
from keras.utils import plot_model
from keras_applications import resnet50

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

from flowcat.visual import plotting
from flowcat.data import case_dataset as cc
from flowcat.data import loaders, all_dataset

from flowcat.models import weighted_crossentropy
from flowcat.models import fcs_cnn, histo_nn, som_cnn, merged_classifiers

from flowcat import utils
from flowcat.configuration import Configuration
from flowcat.mappings import NAME_MAP, GROUP_MAPS

# choose another directory to save downloaded data
if "flowCat_tmp" in os.environ:
    utils.TMP_PATH = os.environ["flowCat_tmp"]

COLS = "grcmyk"

LOGGER = logging.getLogger(__name__)


GLOBAL_DECAY = 0.001 / 2  # division by two for usage in l2 regularization


MODEL_CONSTRUCTORS = {
    "histogram": histo_nn.create_model_histo,
    "som": som_cnn.create_model_cnn,
    "etefcs": fcs_cnn.create_model_fcs,
}


MODEL_BATCH_SIZES = {
    "train": {
        "histogram": 64,
        "som": 16,
        "etefcs": 16,
    },
    "test": {
        "histogram": 128,
        "som": 32,
        "etefcs": 16,
    }
}


def inverse_binarize(y, classes):
    classes = np.asarray(classes)
    if isinstance(y, pd.DataFrame):
        y = y.values
    if len(classes) > 2:
        return classes.take(y.argmax(axis=1), mode="clip")
    elif len(classes) == 2:
        return classes.take(y[:, 1].astype("int64"))
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


def run_save_model(
        model, trainseq, testseq,
        train_epochs=200, epsilon=1e-8, initial_rate=1e-3, drop=0.5, epochs_drop=100, num_workers=1, validation=False,
        weights=None, path="mll-sommaps/models", name="0"):
    """Run and predict using the given model. Also save the model in the given
    path with specified name."""

    # save the model weights after training
    modelpath = pathlib.Path(path)
    modelpath.mkdir(parents=True, exist_ok=True)

    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights.values)

        weights.to_csv(modelpath / f"weights_{name}.csv")
        plotting.plot_confusion_matrix(
            weights.values, weights.columns, normalize=False, cmap=cm.get_cmap("Reds"),
            title="Weight Matrix",
            filename=modelpath / f"weightsplot_{name}", dendroname=None)

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

    # plot a model diagram
    keras.utils.plot_model(model, modelpath / f"modelplot_{name}.png", show_shapes=True)

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
        use_multiprocessing=True,
    )
    pred_mat = model.predict_generator(testseq, workers=num_workers, use_multiprocessing=True)

    model.save(str(modelpath / f"model_{name}.h5"))
    with open(str(modelpath / f"history_{name}.p"), "wb") as hfile:
        pickle.dump(history.history, hfile)

    trainhistory_path = modelpath / f"trainhistory_{name}"
    plot_train_history(trainhistory_path, history.history)

    pred_df = pd.DataFrame(
        pred_mat, columns=testseq.groups, index=testseq.labels)
    pred_df["correct"] = testseq.ylabels
    pred_df.to_csv(modelpath / f"predictions_{name}.csv")
    return pred_df


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
        train_data, test_data, name, loader_options, traindata_args, testdata_args, path=None):
    """Create model and associated inputs.
    Args:
        train_data, test_data: Dataset used to infer sizes.
        name: String name of model.
        loader_options: Options for specific file loader types.
        traindata_args: Options for training dataset.
        testdata_args: Options for test dataset.
        path: Optional path to load previous model from.
    Returns:
        Tuple of model and partial dataset constructor functions.
    """
    output_spec = create_output_spec(name, loader_options)
    train = loaders.DatasetSequence(train_data, output_spec, **traindata_args)
    test = loaders.DatasetSequence(test_data, output_spec, **testdata_args)
    if path is not None:
        model = keras.models.load_model(path, compile=False)
    else:
        model = MODEL_CONSTRUCTORS[name](*train.shape)

    return model, train, test


def load_dataset(cases, paths, filters, mapping):
    LOGGER.info("Loading combined dataset")
    dataset = all_dataset.CombinedDataset.from_paths(cases, paths)
    dataset.filter(**filters)
    dataset.set_mapping(GROUP_MAPS[mapping])
    dataset.set_available([n for n, _ in paths] + ["FCS"])
    return dataset


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
        plotting.plot_confusion_matrix(
            conf.values, groupstat["groups"], normalize=True,
            title=title, filename=outpath / f"confusion_{gname}.png", dendroname=None, sizes=sizes)
        utils.save_csv(conf, outpath / f"confusion_{gname}.csv")
        with open(str(outpath / f"stats_{gname}.json"), "w") as jsfile:
            json.dump(stats, jsfile)


def setup_logging(logpath):
    # setup logging
    filelog = logging.FileHandler(str(logpath))
    filelog.setLevel(logging.DEBUG)
    printlog = logging.StreamHandler()
    printlog.setLevel(logging.INFO)

    fileform = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    filelog.setFormatter(fileform)
    printform = logging.Formatter("%(levelname)s - %(message)s")
    printlog.setFormatter(printform)

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(filelog)
    LOGGER.addHandler(printlog)
    modulelog = logging.getLogger("flowcat")
    modulelog.setLevel(logging.INFO)
    modulelog.addHandler(filelog)


def create_config():
    # CONFIGURATION VARIABLES
    c_general_name = "etefcs"

    # Output options
    c_output_results = "output/test/output"
    c_output_model = "output/test/models"

    # file locations
    c_dataset_cases = "s3://mll-flowdata/fixedCLL-9F"
    c_dataset_paths = [
        # ("SOM", "output/mll-sommaps/sample_maps/testrun_s32_ttoroid"),
        # ("HISTO", "s3://mll-flow-classification/clustering/abstract/abstract_somgated_1_20180723_1217"),
    ]
    c_dataset_filters = {
        "tubes": [1, 2],
        "counts": 10000,
        "groups": ["CLL", "MBL", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"],
        # "groups": ["CLL", "normal"],
        "num": None,
    }
    # available: 8class 6class 5class 3class 2class
    # see flowcat.mappings for details
    c_dataset_mapping = "2class"

    # specific train test splitting
    c_split_test_num = 0.2
    c_split_train_labels = None
    c_split_test_labels = None

    # load existing model and use different parameters for retraining
    # available models: histogram, som, etefcs
    c_model_name = "etefcs"
    c_model_loader_options = {
        loaders.FCSLoader.__name__: {
            "subsample": 200,
            "randomize": False,  # Always change the subsample in different epochs, MUCH SLOWER!
        },
        loaders.CountLoader.__name__: {
            "datatype": "dataframe",
        },
        loaders.Map2DLoader.__name__: {
            "sel_count": "counts",  # use count in SOM map as just another channel
            "pad_width": 1,  # TODO: infer from dataset paths
        },
    }
    c_model_traindata_args = {
        "batch_size": MODEL_BATCH_SIZES["train"][c_model_name],
        "draw_method": "shuffle",  # possible: sequential, shuffle, balanced, groupnum
        "epoch_size": None,
        "sample_weights": False,
    }
    c_model_testdata_args = {
        "batch_size": MODEL_BATCH_SIZES["test"][c_model_name],
        "draw_method": "sequential",
        "epoch_size": None,
    }
    c_model_path = None  # load model weights from the given path

    # Run options
    c_run_weights = None
    c_run_train_epochs = 100
    c_run_initial_rate = 1e-4
    c_run_drop = 0.5
    c_run_epochs_drop = 50
    c_run_epsilon = 1e-8
    c_run_num_workers = 1
    c_run_validation = False
    c_run_path = f"{c_output_model}/{c_general_name}"

    # Stat output
    c_stat_confusion_sizes = True
    # END CONFIGURATION VARIABLES
    # save configuration variables
    config = Configuration.from_localsdict(locals())
    return config

def main():
    parser = argparse.ArgumentParser(description="Classify samples")
    parser.add_argument("--seed", help="Seed for random number generator", type=int)
    parser.add_argument("--config", help="Path to configuration file.", type=utils.URLPath)
    parser.add_argument("--recreate", help="Recreate existing files.", action="store_true")
    args = parser.parse_args()

    if args.config:
        config = Configuration.from_file(args.config)
    else:
        config = create_config()

    outpath = utils.URLPath(f"{config['output']['results']}/{config['general']['name']}")

    # Create logfiles
    logpath = outpath / f"classification.log"
    logpath.local.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(logpath)

    # save configuration
    config.to_toml(outpath / "config.toml")

    dataset = load_dataset(**config["dataset"])

    # split into train and test set
    LOGGER.info("Splitting dataset")
    train, test = all_dataset.split_dataset(dataset, **config["split"])
    utils.save_json(train.labels, outpath / "train_labels.json", clobber=args.recreate)
    utils.save_json(test.labels, outpath / "test_labels.json", clobber=args.recreate)

    LOGGER.info("Getting models")
    model, trainseq, testseq = generate_model_inputs(train, test, **config["model"])

    LOGGER.info("Running model")
    pred_df = run_save_model(
        model, trainseq, testseq, name="0", **config["run"])

    LOGGER.info("Creating statistics")
    create_stats(
        outpath=outpath, dataset=dataset, pred_df=pred_df, **config["stat"])


if __name__ == "__main__":
    main()
