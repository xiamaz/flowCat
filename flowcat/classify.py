"""
Classification functions.
"""
import math
import logging

import numpy as np
import pandas as pd

from matplotlib import cm

from sklearn import metrics

import keras
from keras import models, optimizers
from keras.utils import plot_model

from . import configuration, utils, mappings, loaders, som
from .dataset import combined_dataset
from .visual import classify_plots, plotting
from .models import weighted_crossentropy, classifiers


LOGGER = logging.getLogger(__name__)


class SOMClassifierConfig(configuration.ClassificationConfig):
    """Generate configuration for SOM classification"""

    name = "modelconfig"
    desc = "SOM classifier configuration"
    default = ""

    @classmethod
    def generate_config(cls, args=None):

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
            "fit": {
                "train_epochs": 100,
                "validation": False,
                "pregenerate": True,
            }
        }
        return cls(data)


def compile_model(model, weights=None, epsilon=1e-8, config=None):
    """Compile model."""
    if config is not None:
        weights = config["weights"]
        epsilon = config["epsilon"]

    weights = load_weights(weights)
    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights.values)

    def top2_acc(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    model.compile(
        loss=lossfun,
        # optimizer="adam",
        optimizer=optimizers.Adam(lr=0.0, decay=0.0, epsilon=epsilon),  # lr and decay set by callback
        metrics=[
            "acc",
            # top2_acc,
        ]
    )

    return model


def fit(
        model, data, data_val=None, train_epochs=10,
        initial_rate=1e-3, drop=0.5, epochs_drop=100,
        num_workers=1, pregenerate=True, config=None):

    if config is not None:
        train_epochs = config["train_epochs"]
        initial_rate = config["initial_rate"]
        drop = config["drop"]
        epochs_drop = config["epochs_drop"]
        num_workers = config["num_workers"]
        pregenerate = config["pregenerate"]

    def create_stepped(initial_rate=1e-3, drop=0.5, epochs_drop=100):

        def scheduler(epoch):
            lrate = initial_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        return keras.callbacks.LearningRateScheduler(scheduler)

    if pregenerate:
        data.generate_batches(num_workers)
        if data_val is not None:
            data_val.generate_batches(num_workers)
        LOGGER.info("Pregenerating batches in sequences. Worker number will be ignored in the fit generator")
        num_workers = 0

    history = model.fit_generator(
        data, epochs=train_epochs,
        callbacks=[
            # keras.callbacks.EarlyStopping(min_delta=0.01, patience=20, mode="min"),
            create_stepped(initial_rate, drop, epochs_drop),
        ],
        validation_data=data_val,
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
    return history


def predict_generator(model, data, num_workers=0, config=None):
    if config is not None:
        num_workers = config["num_workers"]

    pred_mat = model.predict_generator(data, workers=num_workers, use_multiprocessing=True)

    pred_df = pd.DataFrame(pred_mat, columns=data.groups, index=data.labels)
    pred_df["correct"] = data.ylabels
    return pred_df


def predict(model, data, groups, labels, ylabels=None):
    pred_mat = model.predict(data)
    pred_df = pd.DataFrame(pred_mat, columns=groups, index=labels)
    if ylabels is not None:
        pred_df["correct"] = ylabels
    return pred_df


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


def save_model(model, sequence, config, path, history=None, weights=None, dataset=None):
    """Save the given model to the given path.

    This will save the model including all informations needed to load the model
    again and use it to train new data.

    Args:
        model: Model to be saved
        path: Output path to save model in.
    """
    path = utils.URLPath(path)
    # plot a model diagram
    plotpath = path / "modelplot.png"
    plotpath.put(lambda p: keras.utils.plot_model(model, p, show_shapes=True))

    modelpath = path / "model.h5"
    modelpath.put(lambda p: model.save(p))

    if history is not None:
        histpath = path / "history"
        histpath.put(lambda p: classify_plots.plot_train_history(p, history.history))
        utils.save_pickle(history.history, histpath / "history.p")

    if weights is not None:
        save_weights(weights, path)

    if dataset is not None:
        # get configs for each dtype
        for dtype, dset in dataset.datasets.items():
            dset.save_config(path / dtype.name)

    config.to_file(path / "config.toml")

    # Save dataset to a folder with name dataset
    datasetpath = path / "dataset"
    seqconf, outputconfs = sequence.get_config()
    seqconf.to_file(datasetpath / "config.toml")
    for name, sconfig in outputconfs.items():
        sconfig.to_file(datasetpath / f"{name}.toml")


def load_som_model(path):
    config = configuration.SOMConfig.from_file(path / "config.toml")
    reference = som.load_som(path / "reference", tubes=config("dataset", "filters", "tubes"), suffix=True)

    # copy fitmap args into normal args
    fitmap_args = config.data["somnodes"]["fitmap_args"]
    config.data["tfsom"]["max_epochs"] = fitmap_args["max_epochs"]
    config.data["tfsom"]["initial_learning_rate"] = fitmap_args["initial_learn"]
    config.data["tfsom"]["end_learning_rate"] = fitmap_args["end_learn"]
    config.data["tfsom"]["initial_radius"] = fitmap_args["initial_radius"]
    config.data["tfsom"]["end_radius"] = fitmap_args["end_radius"]

    def create_som(case):
        return som.create_som([case], config, reference=reference)

    return create_som


def load_model(path):
    """Load a model from the given directory."""
    path = utils.URLPath(path)
    # load configuration
    config = configuration.ClassificationConfig.from_file(path / "config.toml")
    dataset_config, output_configs = loaders.load_dataset_config(path / "dataset")

    # load model
    model = keras.models.load_model(str((path / "model.h5").get()), compile=False)
    compile_model(model, config=config("model", "compile"))

    # load dataset loaders
    sommodel = load_som_model(path / "SOM")

    dataseq = loaders.DatasetSequence.from_config(dataset_config, output_configs)

    def transform(cases):
        """Build a transformer closure to transform data into correct format from single cases."""
        result = [[] for i in range(len(dataseq.output_dtypes))]
        for case in cases:
            somweights = sommodel(case)
            transformed = dataseq.transform(somweights)
            for i, tdata in enumerate(transformed):
                result[i].append(tdata)
        return result
    return model, transform, dataseq.groups


def save_predictions(df, path):
    """Save the given predictions into a specified csv file."""
    predpath = path / f"predictions.csv"
    predpath.put(lambda p: df.to_csv(str(p)))


def load_weights(weights):
    """Load weights using named labels or load matrices from paths."""
    # TODO implement real functionality
    if weights is not None:
        raise RuntimeError("Weights have not been implemented yet.")
    return weights


def save_weights(weights, path, plot=False):
    """Save weighted matrix."""
    if weights is not None:
        weights.to_csv(path / f"weights.csv")
        plotting.plot_confusion_matrix(
            weights.values, weights.columns, normalize=False, cmap=cm.get_cmap("Reds"),
            title="Weight Matrix",
            filename=path / f"weightsplot", dendroname=None)


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


def generate_model_inputs(train_data, test_data, config):
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

    LOGGER.debug("Creating output spec for model %s.", mtype)
    output_spec = create_output_spec(mtype, loader_options)

    LOGGER.debug("Creating train and test data sequences")
    train = loaders.DatasetSequence.from_data(train_data, output_spec, **traindata_args)
    test = loaders.DatasetSequence.from_data(test_data, output_spec, **testdata_args)

    model = classifiers.mtype_to_model(mtype)(*train.shape)

    model = compile_model(model, config=config["compile"])

    return model, train, test


def create_stats(outpath, dataset, pred_df, confusion_sizes):
    LOGGER.info("=== Statistics results ===")
    for gname, groupstat in mappings.GROUP_MAPS.items():
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
