import math
import json
import os
import pickle
import pathlib
import functools
import logging

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
from flowcat.data import loaders

from flowcat.models import weighted_crossentropy
from flowcat.models import fcs_cnn, histo_nn, som_cnn, merged_classifiers

from flowcat import utils
from flowcat.configuration import Configuration
from flowcat.mappings import NAME_MAP, GROUP_MAPS, PATHOLOGIC_NORMAL

# choose another directory to save downloaded data
if "flowCat_tmp" in os.environ:
    utils.TMP_PATH = os.environ["flowCat_tmp"]

COLS = "grcmyk"

LOGGER = logging.getLogger(__name__)


GLOBAL_DECAY = 0.001 / 2  # division by two for usage in l2 regularization


def inverse_binarize(y, classes):
    classes = np.asarray(classes)
    if isinstance(y, pd.DataFrame):
        y = y.values
    if len(classes) > 2:
        return classes.take(y.argmax(axis=1), mode="clip")

    raise RuntimeError("Inverse binary not implemented yet.")


def get_tubepaths(label, cases):
    matched = None
    for case in cases:
        if case.id == label:
            matched = case
            break
    tubepaths = {k: v[-1].path for k, v in matched.tubepaths.items()}
    return tubepaths


def load_histolabels(histopath):
    counts = []
    for tube in [1, 2]:
        df = pd.read_csv(
            f"{histopath}/tube{tube}.csv", index_col=0
        )
        df["group"] = df["group"].apply(lambda g: NAME_MAP.get(g, g))
        df.set_index(["label", "group"], inplace=True)
        count = pd.DataFrame(1, index=df.index, columns=["count"])
        counts.append(count)
    both_labels = functools.reduce(lambda x, y: x.add(y, fill_value=0), counts)
    return both_labels


def rescale_sureness(data):
    data["sureness"] = data["sureness"] / data["sureness"].mean() * 5
    return data


def create_dataset(som=None, histo=None, fcs=None):
    """Creata a new dataset table."""
    mappath = pathlib.Path(som)
    sommap_labels = pd.read_csv(f"{mappath}.csv", index_col=0).set_index(["label", "group"])
    sommap_count = pd.DataFrame(1, index=sommap_labels.index, columns=["count"])
    histo_count = load_histolabels(histo)
    both_count = sommap_count.add(histo_count, fill_value=0)
    # both_count = both_count.loc[both_count["count"] == 3, :]

    assert not both_count.empty, "No data having both histo and sommap info."

    cdict = {}
    cases = cc.CaseCollection(fcs, tubes=[1, 2])
    caseview = cases.filter(counts=10000)
    for case in caseview:
        material = case.has_same_material([1, 2])
        fcspaths = {t: str(case.get_tube(t, material=material, min_count=10000).path.local) for t in [1, 2]}
        try:
            assert both_count.loc[(case.id, case.group), "count"] == 3, "Not all data available."
            cdict[case.id] = {
                "group": case.group,
                "sommappath": str(mappath / f"{case.id}_t{{tube}}.csv"),
                "fcspath": fcspaths,
                "histopath": f"{histo}/tube{{tube}}.csv",
                "sureness": case.sureness,
            }
        except KeyError as e:
            LOGGER.debug(f"{e} - Not found in histo or sommap")
            continue
        except AssertionError as e:
            LOGGER.debug(f"{case.id}|{case.group} - {e}")
            continue

    dataset = pd.DataFrame.from_dict(cdict, orient="index")

    # scale sureness to mean 5 per group
    dataset = dataset.groupby("group").apply(rescale_sureness)
    return dataset


def load_dataset(index=None, paths=None, mapping=None):
    """Return dataframe containing columns with filename and labels.
    Args:
    """
    if index is None:
        dataset = create_dataset(**paths)
    else:
        dataset = utils.load_pickle(utils.URLPath(index))

    mapdict = GROUP_MAPS[mapping]
    dataset = dataset_apply_mapping(dataset, mapdict)
    return dataset, mapdict


def dataset_apply_mapping(dataset, mapping):
    """Apply a specific mapping to the given dataset."""
    # copy group into another column
    dataset["orig_group"] = dataset["group"]
    if mapping is not None:
        dataset = modify_groups(dataset, mapping=mapping["map"])
        dataset = dataset.loc[dataset["group"].isin(mapping["groups"]), :]
    return dataset


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
    ax.plot(
        range(len(data["loss"])), data["val_loss"],
        c="red", linestyle="--", label="Validation Loss")

    # Testing dataset loss and accuracy metrics
    ax.plot(
        range(len(data["acc"])), data["acc"],
        c="blue", linestyle="-", label="Accuracy")
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
        train_epochs=200, epsilon=1e-8, initial_rate=1e-3, drop=0.5, epochs_drop=100,
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
            weights.values, weights.columns, normalize=False, cmap=cm.Reds,
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
        validation_data=testseq,
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
        workers=4,
        use_multiprocessing=True,
    )
    pred_mat = model.predict_generator(testseq, workers=4, use_multiprocessing=True)

    model.save(modelpath / f"model_{name}.h5")
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


def modify_groups(data, mapping):
    """Change the cohort composition according to the given
    cohort composition."""
    data["group"] = data["group"].apply(lambda g: mapping.get(g, g))
    return data


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


def split_data(data, test_num=None, test_labels=None, train_labels=None):
    """Split data in stratified fashion by group.
    Args:
        data: Dataset to be split. Label should be contained in 'group' column.
        test_num: Ratio of samples in test per group or absolute number of samples in each group for test.
    Returns:
        (train, test) with same columns as input data.
    """
    if test_labels is not None:
        if not isinstance(test_labels, list):
            test_labels = utils.load_json(test_labels)
        test = data.loc[test_labels, :]
    if train_labels is not None:
        if not isinstance(train_labels, list):
            train_labels = utils.load_json(train_labels)
        train = data.loc[train_labels, :]
    if test_num is not None:
        assert test_labels is None and train_labels is None, "Cannot use num with specified labels"
        grouped = data.groupby("group")
        if test_num < 1:
            test = grouped.apply(lambda d: d.sample(frac=test_num)).reset_index(level=0, drop=True)
        else:
            group_sizes = grouped.size()
            if any(group_sizes <= test_num):
                insuff = group_sizes[group_sizes <= test_num]
                LOGGER.warning("Insufficient sizes: %s", insuff)
                raise RuntimeError("Some cohorts are too small.")
            test = grouped.apply(lambda d: d.sample(n=test_num)).reset_index(level=0, drop=True)
        train = data.drop(test.index, axis=0)
    return train, test


def create_output_spec(modelname, dataoptions):
    if modelname == "histogram":
        partial_spec = [
            (loaders.CountLoader.create_inferred, {"tube": 1}),
            (loaders.CountLoader.create_inferred, {"tube": 2}),
        ]
    elif modelname == "sommap":
        partial_spec = [
            (loaders.Map2DLoader.create_inferred, {"tube": 1}),
            (loaders.Map2DLoader.create_inferred, {"tube": 2}),
        ]
    elif modelname == "maphisto":
        partial_spec = [
            (loaders.Map2DLoader.create_inferred, {"tube": 1}),
            (loaders.Map2DLoader.create_inferred, {"tube": 2}),
            (loaders.CountLoader.create_inferred, {"tube": 1}),
            (loaders.CountLoader.create_inferred, {"tube": 2}),
        ]
    elif modelname == "etefcs":
        partial_spec = [
            (loaders.FCSLoader.create_inferred, {"tubes": [1, 2]}),
        ]
    elif modelname == "mapfcs":
        partial_spec = [
            (loaders.FCSLoader.create_inferred, {"tubes": [1, 2]}),
            (loaders.Map2DLoader.create_inferred, {"tube": 1}),
            (loaders.Map2DLoader.create_inferred, {"tube": 2}),
        ]
    elif modelname == "maphistofcs":
        partial_spec = [
            (loaders.FCSLoader.create_inferred, {"tubes": [1, 2]}),
            (loaders.Map2DLoader.create_inferred, {"tube": 1}),
            (loaders.Map2DLoader.create_inferred, {"tube": 2}),
            (loaders.CountLoader.create_inferred, {"tube": 1}),
            (loaders.CountLoader.create_inferred, {"tube": 2}),
        ]
    else:
        raise RuntimeError(f"Unknown model {modelname}")

    # create partial loader functions
    output_spec = [loaders.loader_builder(f, **{**v, **dataoptions[f.__self__.__name__]}) for f, v in partial_spec]

    return output_spec


MODEL_CONSTRUCTORS = {
    "histogram": histo_nn.create_model_histo,
    "sommap": som_cnn.create_model_cnn,
    "maphisto": merged_classifiers.create_model_maphisto,
    "etefcs": fcs_cnn.create_model_fcs,
    "mapfcs": merged_classifiers.create_model_mapfcs,
    "maphistofcs": merged_classifiers.create_model_all,
}


MODEL_BATCH_SIZES = {
    "train": {
        "histogram": 64,
        "sommap": 16,
        "maphisto": 32,
        "etefcs": 16,
        "mapfcs": 16,
        "maphistofcs": 32,
    },
    "test": {
        "histogram": 128,
        "sommap": 32,
        "maphisto": 32,
        "etefcs": 16,
        "mapfcs": 32,
        "maphistofcs": 32,
    }
}


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
    modulelog = logging.getLogger("clustering")
    modulelog.setLevel(logging.INFO)
    modulelog.addHandler(filelog)


def main():
    # CONFIGURATION VARIABLES
    c_general_name = "newconfig"

    # Output options
    c_output_results = "mll-sommaps/output"
    c_output_model = "mll-sommaps/models"
    c_output_confusion_sizes = True

    # file locations
    c_dataset_index = None  # use pregenerated dataindex instead
    c_dataset_paths = {
        "som": "mll-sommaps/sample_maps/selected1_toroid_s32",
        "histo": "../mll-flow-classification/clustering/abstract/abstract_somgated_1_20180723_1217",
        "fcs": "s3://mll-flowdata/CLL-9F",
    }
    c_dataset_mapping = "8class"

    # specific train test splitting
    c_split_test_num = None
    c_split_train_labels = "data/train_labels.json"
    c_split_test_labels = "data/test_labels.json"

    # load existing model and use different parameters for retraining
    c_model_name = "sommap"
    c_model_loader_options = {
        loaders.FCSLoader.__name__: {
            "subsample": 100,
        },
        loaders.CountLoader.__name__: {
            "version": "dataframe",
        },
        loaders.Map2DLoader.__name__: {
            "sel_count": "counts",  # use count in SOM map as just another channel
            "pad_width": 1 if "toroid" in c_dataset_paths["som"] else 0  # do something more elaborate later
        },
    }
    c_model_traindata_args = {
        "batch_size": MODEL_BATCH_SIZES["train"][c_model_type],
        "draw_method": "balanced",
        "epoch_size": 16000,
        "sample_weights": True,
    }
    c_model_testdata_args = {
        "batch_size": MODEL_BATCH_SIZES["test"][c_model_type],
        "draw_method": "sequential",
        "epoch_size": None,
    }
    c_model_path = None

    # Run options
    c_run_weights = None
    c_run_train_epochs = 100
    c_run_initial_rate = 1e-4
    c_run_drop = 0.5
    c_run_epochs_drop = 50
    c_run_epsilon = 1e-8
    # END CONFIGURATION VARIABLES

    # save configuration variables
    config = Configuration.from_localsdict(locals())

    outpath = utils.URLPath(f"{c_output_results}/{config['general']['name']}")
    modelpath = utils.URLPath(f"{c_output_model}/{config['general']['name']}")

    # Create logfiles
    logpath = outpath / f"classification.log"
    setup_logging(logpath)

    # save configuration
    config.to_toml(outpath / "config.toml")

    dataset, mapping = load_dataset(**config["dataset"])
    # save it if newly generated
    if config["dataset"]["index"] is None:
        utils.save_pickle(dataset, outpath / "dataset.p")

    # split into train and test set
    # TODO: enable more complicated designs with kfold etc
    train, test = split_data(**config["split"])
    utils.save_json(list(train.index), outpath / "train_labels.json")
    utils.save_json(list(test.index), outpath / "test_labels.json")

    # TODO: save inputspec with saved model for easier loading
    model, trainseq, testseq = generate_model_inputs(train, **config["model"])

    pred_df = run_save_model(
        model, trainseq, testseq, path=modelpath, name="0", **config["run"])

    LOGGER.info("Statistics results for %s", name)
    for gname, groupstat in GROUP_MAPS.items():
        # skip if our cohorts are larger
        if len(mapping["groups"]) < len(groupstat["groups"]):
            continue

        LOGGER.info(f"-- {len(groupstat['groups'])} --")
        conf, stats = create_metrics_from_pred(pred_df, mapping=groupstat["map"])

        # choose whether to represent merged classes larger in confusion matrix
        sizes = groupstat["sizes"] if c_confusion_sizes else None

        title = f"Confusion matrix (weighted f1 {stats['weighted_f1']:.2f} unweighted f1 {stats['unweighted_f1']:.2f})"
        plotting.plot_confusion_matrix(
            conf.values, groupstat["groups"], normalize=True,
            title=title, filename=outpath / f"confusion_{gname}", dendroname=None, sizes=sizes)
        conf.to_csv(outpath / f"confusion_{gname}.csv")
        with open(str(outpath / f"stats_{gname}.json"), "w") as jsfile:
            json.dump(stats, jsfile)


if __name__ == "__main__":
    main()
