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
from flowcat.data import collection as cc
from flowcat.data import loaders

from flowcat.models import weighted_crossentropy
from flowcat.models import fcs_cnn, histo_nn, som_cnn, merged_classifiers

from flowcat import utils
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


def load_dataset(mappath, histopath, fcspath):
    """Return dataframe containing columns with filename and labels."""
    mappath = pathlib.Path(mappath)

    sommap_labels = pd.read_csv(f"{mappath}.csv", index_col=0).set_index(["label", "group"])
    sommap_count = pd.DataFrame(1, index=sommap_labels.index, columns=["count"])
    histo_count = load_histolabels(histopath)
    both_count = sommap_count.add(histo_count, fill_value=0)
    # both_count = both_count.loc[both_count["count"] == 3, :]

    assert not both_count.empty, "No data having both histo and sommap info."

    cdict = {}
    cases = cc.CaseCollection(fcspath, tubes=[1, 2])
    caseview = cases.create_view(counts=10000)
    for case in caseview:
        try:
            assert both_count.loc[(case.id, case.group), "count"] == 3, "Not all data available."
            cdict[case.id] = {
                "group": case.group,
                "sommappath": str(mappath / f"{case.id}_t{{tube}}.csv"),
                "fcspath": {k: utils.get_file_path(v[-1].path) for k, v in case.tubepaths.items()},
                "histopath": f"{histopath}/tube{{tube}}.csv",
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
            test_labels = load_json(test_labels)
        test = data.loc[test_labels, :]
    if train_labels is not None:
        if not isinstance(train_labels, list):
            train_labels = load_json(train_labels)
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


def load_json(jspath):
    with open(str(jspath), "r") as jsfile:
        data = json.load(jsfile)

    return data


def save_json(data, outpath):
    """Save a json file to the specified path."""
    with open(str(outpath), "w") as jsfile:
        json.dump(data, jsfile)


def save_pickle(data, outpath):
    with open(str(outpath), "wb") as pfile:
        pickle.dump(data, pfile)


def load_pickle(path):
    with open(str(path), "rb") as pfile:
        data = pickle.load(pfile)
    return data


def get_model_type(modelname, dataoptions, data):
    if modelname == "histogram":
        xoutputs = [
            loaders.CountLoader.create_inferred(data, tube=1, **dataoptions[loaders.CountLoader.__name__]),
            loaders.CountLoader.create_inferred(data, tube=2, **dataoptions[loaders.CountLoader.__name__]),
        ]
        train_batch = 64
        test_batch = 128
        modelfun = histo_nn.create_model_histo
    elif modelname == "sommap":
        xoutputs = [
            loaders.Map2DLoader.create_inferred(data, tube=1, **dataoptions[loaders.Map2DLoader.__name__]),
            loaders.Map2DLoader.create_inferred(data, tube=2, **dataoptions[loaders.Map2DLoader.__name__]),
        ]
        train_batch = 16
        test_batch = 32
        modelfun = som_cnn.create_model_cnn
    elif modelname == "maphisto":
        xoutputs = [
            loaders.Map2DLoader.create_inferred(data, tube=1, **dataoptions[loaders.Map2DLoader.__name__]),
            loaders.Map2DLoader.create_inferred(data, tube=2, **dataoptions[loaders.Map2DLoader.__name__]),
            loaders.CountLoader.create_inferred(data, tube=1, **dataoptions[loaders.CountLoader.__name__]),
            loaders.CountLoader.create_inferred(data, tube=2, **dataoptions[loaders.CountLoader.__name__]),
        ]
        train_batch = 32
        test_batch = 32
        modelfun = merged_classifiers.create_model_maphisto
    elif modelname == "etefcs":
        xoutputs = [
            loaders.FCSLoader.create_inferred(data, tubes=[1, 2], **dataoptions[loaders.FCSLoader.__name__]),
        ]
        train_batch = 16
        test_batch = 16
        modelfun = fcs_cnn.create_model_fcs
    elif modelname == "mapfcs":
        xoutputs = [
            loaders.FCSLoader.create_inferred(data, tubes=[1, 2], **dataoptions[loaders.FCSLoader.__name__]),
            loaders.Map2DLoader.create_inferred(data, tube=1, **dataoptions[loaders.Map2DLoader.__name__]),
            loaders.Map2DLoader.create_inferred(data, tube=2, **dataoptions[loaders.Map2DLoader.__name__]),
        ]
        train_batch = 32
        test_batch = 32
        modelfun = merged_classifiers.create_model_mapfcs
    elif modelname == "maphistofcs":
        xoutputs = [
            loaders.FCSLoader.create_inferred(data, tubes=[1, 2], **dataoptions[loaders.FCSLoader.__name__]),
            loaders.Map2DLoader.create_inferred(data, tube=1, **dataoptions[loaders.Map2DLoader.__name__]),
            loaders.Map2DLoader.create_inferred(data, tube=2, **dataoptions[loaders.Map2DLoader.__name__]),
            loaders.CountLoader.create_inferred(data, tube=1, **dataoptions[loaders.CountLoader.__name__]),
            loaders.CountLoader.create_inferred(data, tube=2, **dataoptions[loaders.CountLoader.__name__]),
        ]
        train_batch = 32
        test_batch = 32
        modelfun = merged_classifiers.create_model_all
    else:
        raise RuntimeError(f"Unknown model {modelname}")

    return modelfun, xoutputs, train_batch, test_batch


def main():
    ## CONFIGURATION VARIABLES
    c_uniq_name = "relunet_yesglobal_200epoch_sample_weighted1510"
    c_model = "sommap"
    c_groupmap = "8class"
    c_weights = None
    # output locations
    c_output_results = "mll-sommaps/output"
    c_output_model = "mll-sommaps/models"
    # file locations
    c_dataindex = None  # use pregenerated dataindex instead
    c_sommap_data = "mll-sommaps/sample_maps/selected1_toroid_s32"
    c_histo_data = "../mll-flow-classification/clustering/abstract/abstract_somgated_1_20180723_1217"
    c_fcs_data = "s3://mll-flowdata/CLL-9F"
    # load existing model and use different parameters for retraining
    # c_modelpath = "mll-sommaps/models/relunet_globalavg_avgmerge_400epoch_sommap_8class/model_0.h5"
    c_modelpath = None
    c_model_runargs = {
        "train_epochs": 100,
        "initial_rate": 1e-4,
        "drop": 0.5,
        "epochs_drop": 50,
        "epsilon": 1e-8,
    }
    # split train, test using predefined split
    c_predefined_split = True
    c_train_labels = "data/train_labels.json"
    c_test_labels = "data/test_labels.json"
    c_trainargs = {
        "draw_method": "balanced",
        "epoch_size": 16000,
        "sample_weights": True,
    }
    c_testargs = {
        "draw_method": "sequential",
        "epoch_size": None,
    }
    c_runargs = {
        "train_epochs": 200,
        "initial_rate": 1e-3,
        "drop": 0.5,
        "epochs_drop": 50,
        "epsilon": 1e-8,
    }
    # Data modifications
    c_dataoptions = {
        loaders.FCSLoader.__name__: {
            "subsample": 100,
        },
        loaders.CountLoader.__name__: {
            "version": "dataframe",
        },
        loaders.Map2DLoader.__name__: {
            "sel_count": "counts",  # use count in SOM map as just another channel
            "pad_width": 1 if "toroid" in c_sommap_data else 0  # do something more elaborate later
        },
    }
    # Output options
    c_confusion_sizes = True
    ## END CONFIGURATION VARIABLES

    # save configuration variables
    config_dict = locals()

    name = f"{c_uniq_name}_{c_model}_{c_groupmap}"
    if c_weights is not None:
        name += f"_{c_weights}"

    # configure logging
    outpath = pathlib.Path(f"{c_output_results}/{name}")
    outpath.mkdir(parents=True, exist_ok=True)

    # save the currently used configuration
    save_json(config_dict, outpath / "config.json")

    filelog = logging.FileHandler(outpath / f"{name}.log")
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

    if c_dataindex is None:
        indata = load_dataset(mappath=c_sommap_data, histopath=c_histo_data, fcspath=c_fcs_data)
        # save the datainfo into the output folder
        with open(str(outpath / "data_paths.p"), "wb") as f:
            pickle.dump(indata, f)
    else:
        # load the data again
        with open(c_dataindex, "rb") as f:
            indata = pickle.load(f)
    groups = GROUP_MAPS[c_groupmap]["groups"]
    group_map = GROUP_MAPS[c_groupmap]["map"]

    indata["orig_group"] = indata["group"]
    indata = modify_groups(indata, mapping=group_map)
    indata = indata.loc[indata["group"].isin(groups), :]

    if c_weights == "weighted":
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
    elif c_weights == "simpleweights":
        # simpler group weights
        group_weights = {
            ("normal", None): (1.0, 20.0),
        }
        weights = create_weight_matrix(group_weights, groups, base_weight=1)
    elif c_weights == "normalweights":
        group_weights = {
            ("normal", None): (10.0, 10.0),
        }
        weights = create_weight_matrix(group_weights, groups, base_weight=1)
    else:
        weights = None

    if c_predefined_split:
        train, test = split_data(
            indata, train_labels=c_train_labels, test_labels=c_test_labels)
    else:
        train, test = split_data(indata, test_num=0.1)

    # always save the labels used for training and testing
    save_json(list(train.index), outpath / "train_labels.json")
    save_json(list(test.index), outpath / "test_labels.json")

    if "toroidal" in c_sommap_data:
        toroidal = True
    else:
        toroidal = False

    modelpath = pathlib.Path(f"{c_output_model}/{name}")

    # get model input and settings depending on model type
    modelfun, xoutputs, train_batch, test_batch = get_model_type(c_model, c_dataoptions, train)
    # create datasets
    trainseq = loaders.SOMMapDataset(train, xoutputs, batch_size=train_batch, groups=groups, **c_trainargs)
    testseq = loaders.SOMMapDataset(test, xoutputs, batch_size=test_batch, groups=groups, **c_testargs)
    # create model using training data shape
    if c_modelpath is None:
        model = modelfun(*trainseq.shape)
        pred_df = run_save_model(
            model, trainseq, testseq, weights=weights, path=modelpath, name="0", **c_runargs)
    else:
        model = keras.models.load_model(c_modelpath, compile=False)
        pred_df = run_save_model(
            model, trainseq, testseq, weights=weights, path=modelpath, name="0", **c_model_runargs)

    # fit the model

    LOGGER.info("Statistics results for %s", name)
    for gname, groupstat in GROUP_MAPS.items():
        # skip if our cohorts are larger
        if len(groups) < len(groupstat["groups"]):
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
