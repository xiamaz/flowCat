import math
import json
import os
import pickle
import pathlib
import functools
import hashlib
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

import fcsparser

import weighted_crossentropy

import sys
sys.path.append("../classification")
from classification import plotting
from clustering import collection as cc
from clustering import utils

# always put the tmp folder in the home directory for now
utils.TMP_PATH = f"{os.environ['HOME']}/tmp"

CACHEDIR = "cache"

NAME_MAP = {
    "HZL": "HCL",
    "HZLv": "HCLv",
    "Mantel": "MCL",
    "Marginal": "MZL",
    "CLLPL": "PL"
}

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
    return dataset


def normalize_data(data):
    data["data"] = data["data"].apply(
        lambda t: [
            pd.DataFrame(
                preprocessing.MinMaxScaler().fit_transform(d),
                columns=d.columns
            ) for d in t]
    )
    return data


def reshape_dataframe(data, m=10, n=10, pad_width=0):
    """Reshape dataframe 2d matrix with channels as additional dimension.
    Optionally pad the data.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.reshape(data, (m, n, -1))
    if pad_width:
        data = np.pad(data, pad_width=[
            (pad_width, pad_width),
            (pad_width, pad_width),
            (0, 0),
        ], mode="wrap")
    return data


def select_drop_counts(data, sel_count=None):
    """Select and preprocess count channel. If sel_count is None, drop
    all count channels.
    """
    countnames = ["counts", "count_prev"]
    if sel_count is not None:
        data[sel_count] = np.sqrt(data[sel_count])
        # rescale 0-1
        data[sel_count] = data[sel_count] / max(data[sel_count])

    data.drop(
        [c for c in countnames if c != sel_count], axis=1, inplace=True,
        errors="ignore"
    )
    return data


def args_hasher(*args, **kwargs):
    """Use at own discretion. Will simply concatenate all input args as
    strings to generate keys."""
    h = hashlib.blake2b()
    hashstr = "".join(str(a) for a in args) + "".join(str(k) + str(v) for k, v in kwargs.items())
    h.update(hashstr.encode())
    return h.hexdigest()


def disk_cache(fun):

    cachepath = pathlib.Path(CACHEDIR) / fun.__name__
    cachepath.mkdir(parents=True, exist_ok=True)

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        hashed = args_hasher(*args, **kwargs)
        filepath = cachepath / hashed
        if filepath.exists():
            with open(filepath, "rb") as f:
                result = pickle.load(f)
        else:
            result = fun(*args, **kwargs)
            with open(filepath, "wb") as f:
                pickle.dump(result, f)
        return result

    return wrapper


def mem_cache(fun):
    """Cache function output inside the calling object."""

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_cache"):
            self._cache = {}
        hashed = args_hasher(*args, **kwargs)
        if hashed in self._cache:
            result = self._cache[hashed]
        else:
            result = fun(self, *args, **kwargs)
            self._cache[hashed] = result
        return result

    return wrapper

class LoaderMixin:
    datacol = "sommappath"
    histocol = "histopath"
    fcscol = "fcspath"

    @staticmethod
    def load_data(path, tube):
        """Load the data associated with the given tube."""
        return pd.read_csv(str(path).format(tube=tube), index_col=0)


class CountLoader(LoaderMixin):

    """Load count information from 1d histogram analysis."""
    def __init__(self, tube, width, version="mapcount", data=None):
        self.tube = tube
        self.version = version
        self.data = data
        self.width = width

    @staticmethod
    def read_dataframe(path, tube):
        dfdata = pd.read_csv(path.format(tube=tube), index_col=0)
        dfdata["group"] = dfdata["group"].apply(lambda g: NAME_MAP.get(g, g))
        dfdata.set_index(["label", "group"], inplace=True)
        non_number_cols = [c for c in dfdata.columns if not c.isdigit()]
        dfdata.drop(non_number_cols, inplace=True, axis=1)
        return dfdata

    @classmethod
    def create_inferred(cls, data, tube, version="mapcount"):
        dfdata = None
        if version == "dataframe":
            dfdata = cls.read_dataframe(data[cls.histocol].iloc[0], tube=tube)
            width = dfdata.shape[1]
        elif version == "mapcount":
            mapdata = self.load_data(data[cls.datacol].iloc[0], tube)
            width = mapdata.shape[0]
        return cls(tube=tube, width=width, version=version, data=dfdata)

    @property
    def shape(self):
        return (self.width, )

    def __call__(self, data):
        if self.version == "mapcount":
            count_list = []
            for path in data[self.datacol]:
                mapdata = self.load_data(path, self.tube)
                countdata = select_drop_counts(
                    mapdata, sel_count="count_prev")["count_prev"]
                count_list.append(countdata.values)
            counts = np.stack(count_list)
        elif self.version == "dataframe":
            if self.data is None:
                self.data = self.read_dataframe(
                    data[self.datacol].iloc[0], self.tube)
            label_group = [x for x in zip(data.index, data["orig_group"])]
            sel_rows = self.data.loc[label_group, :]
            missing = sel_rows.loc[sel_rows["1"].isna(), :]
            if not missing.empty:
                LOGGER.error(missing)
                raise RuntimeError()
            counts = sel_rows.values
        return counts



class FCSLoader(LoaderMixin):
    """Directly load FCS data associated with the given ids."""
    def __init__(self, tubes, channels=None, subsample=200):
        self.tubes = tubes
        self.channels = channels
        self.subsample = subsample

    @staticmethod
    def hash_input(path, subsample, tubes, channels):
        """Hash the given information to generate ids for caching."""
        h.update(str(path) + str(tube) + str(channels) + str(subsample))
        hashed = h.hexdigest()
        return hashed

    @classmethod
    def create_inferred(cls, data, tubes, subsample=200, channels=None, *args, **kwargs):
        testdata = cls._load_data(
            data[cls.fcscol].iloc[0], subsample, tubes=tubes, channels=channels
        )
        channels = list(testdata.columns)
        return cls(tubes=tubes, channels=channels, subsample=subsample, *args, **kwargs)

    @property
    def shape(self):
        return (self.subsample * len(self.tubes), len(self.channels))

    @staticmethod
    @disk_cache
    def _load_data(pathdict, subsample, tubes, channels=None):
        datas = []
        for tube in tubes:
            _, data = fcsparser.parse(pathdict[tube], data_set=0, encoding="latin-1")

            data.drop([c for c in data.columns if "nix" in c], axis=1, inplace=True)

            data = pd.DataFrame(
                preprocessing.MinMaxScaler().fit_transform(
                    preprocessing.StandardScaler().fit_transform(data)),
                columns=data.columns)

            data = data.sample(n=subsample)

            cols = [c+s for c in data.columns for s in ["", "sig"]]
            sig_cols = [c for c in cols if c.endswith("sig")]
            data = pd.concat(
                [data, pd.DataFrame(1, columns=sig_cols, index=data.index)], axis=1)
            data = data.loc[:, cols]
            datas.append(data)

        merged = pd.concat(datas, sort=False)
        merged = merged.fillna(0)
        if channels:
            return merged[channels].values
        else:
            return merged

    def __call__(self, data):
        mapped_fcs = []
        for path in data[self.fcscol]:
            mapped_fcs.append(self._load_data(path, self.subsample, self.tubes, self.channels))
        return np.stack(mapped_fcs)


class Map2DLoader(LoaderMixin):
    """2-Dimensional SOM maps for 2D-Convolutional processing."""
    def __init__(self, tube, gridsize, channels, sel_count=None, pad_width=0, cached=False):
        """Object to transform input rows into the specified format."""
        self.tube = tube
        self.gridsize = gridsize
        self.channels = channels
        self.sel_count = sel_count
        self.pad_width = pad_width
        self._cache = {}

    @classmethod
    def create_inferred(cls, data, tube, *args, **kwargs):
        """Create with inferred information."""
        return cls(tube=tube, *args, **cls.infer_size(data, tube), **kwargs)

    @classmethod
    def infer_size(cls, data, tube=1):
        """Infer size of input from data."""
        refdata = cls.load_data(data[cls.datacol].iloc[0], tube)
        nodes, channels = refdata.shape
        gridsize = int(np.ceil(np.sqrt(nodes)))
        non_count_channels = [c for c in refdata.columns if "count" not in c]
        return {"gridsize": gridsize, "channels": non_count_channels}

    @property
    def shape(self):
        return (
            self.gridsize + self.pad_width * 2,
            self.gridsize + self.pad_width * 2,
            len(self.channels) + bool(self.sel_count)
        )

    @staticmethod
    @disk_cache
    def _load_sommap(path, tube, sel_count, pad_width, gridsize):
        mapdata = Map2DLoader.load_data(path, tube)
        mapdata = select_drop_counts(mapdata, sel_count)
        mapdata = reshape_dataframe(mapdata, m=gridsize, n=gridsize, pad_width=pad_width)
        return mapdata

    def _get_mapdata(self, pathlist):
        map_list = []
        for path in pathlist:
            data = self._load_sommap(path, self.tube, self.sel_count, self.pad_width, self.gridsize)
            map_list.append(data)
        return np.stack(map_list)

    def __call__(self, data):
        """Output specified format."""
        return self._get_mapdata(list(data[self.datacol]))


class SOMMapDataset(LoaderMixin, keras.utils.Sequence):
    """Dataset for creating and yielding batches of data for keras model.

    Data can be generated by random draw or alternatively in sequence.
    """

    def __init__(
            self, data, xoutputs,
            batch_size=32, draw_method="shuffle", epoch_size=None,
            groups=None, group_nums=None
    ):
        """
        Args:
            data: DataFrame containing labels and paths to data.
            xoutputs: List of output generator objects taking batches of the filepath dataframe.
            batch_size: Number of cases in a single batch.
            draw_method: Method to select cases in a single batch.
                valid: [
                    'shuffle', # shuffle all data and return batches
                    'sequential',  # return data in sequence
                    'balanced'  # present balanced representation of data in one epoch
                    'groupnums'  # specified number of samples per group
                ]
            epoch_size: Number of samples in a single epoch. Is data length if None or 0.
            groups: List of groups to transform the labels into binary matrix.
            group_nums: Number of samples per group for balanced sampling. If
                not given, will evenly distribute the epoch size among all groups.
        Returns:
            SOMMapDataset object.
        """
        self.batch_size = batch_size
        self.draw_method = draw_method

        self._all_data = data
        if groups is None:
            groups = list(self._all_data["group"].unique())
        self.groups = groups
        self.group_nums = group_nums

        self._data = self._sample_data(data, epoch_size)
        self.epoch_size = self._data.shape[0]

        self._xoutputs = xoutputs

    def _sample_data(self, data, epoch_size=None):
        if self.draw_method == "shuffle":
            selection = data.sample(frac=1)
        elif self.draw_method == "sequential":
            selection = data
        elif self.draw_method == "balanced":
            sample_num = int((epoch_size or len(data)) / len(self.groups))
            selection = data.groupby("group").apply(lambda x: x.sample(
                n=sample_num, replace=True)).reset_index(0, drop=True).sample(frac=1)
        elif self.draw_method == "groupnum":
            selection = data.groupby("orig_group").apply(lambda x: x.sample(
                n=self.group_nums[x.name], replace=True)).reset_index(0, drop=True).sample(frac=1)
        else:
            raise RuntimeError(
                f"Unknown draw method: {self.draw_method}. "
                "Valid options are: ['shuffle', 'sequential', 'balanced']")
        return selection

    @property
    def xshape(self):
        """Return shape of xvalues. Should be a list of shapes describing each input.
        """
        return [x.shape for x in self._xoutputs]

    @property
    def yshape(self):
        """Return shape of yvalues. If we only have 2 classes use simple
        binary encoding.
        """
        return len(self.groups)

    @property
    def shape(self):
        """Return tuple of xshape and yshape."""
        return self.xshape, self.yshape

    @property
    def labels(self):
        return self._data.index.values

    @property
    def ylabels(self):
        return self._data["group"]

    def __len__(self):
        """Return the number of batches generated."""
        return int(np.ceil(self.epoch_size / float(self.batch_size)))

    def __getitem__(self, idx):
        """Get a single batch by id."""
        batch_data = self._data.iloc[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        xdata = [x(batch_data) for x in self._xoutputs]

        ydata = batch_data["group"]
        ybinary = preprocessing.label_binarize(ydata, classes=self.groups)
        return xdata, ybinary

    def on_epoch_end(self):
        """Randomly reshuffle data after end of epoch."""
        self._data = self._sample_data(self._all_data, self.epoch_size)


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


def decomposition(dataset):
    """Non-linear decompositions of the data for visualization purposes."""
    t1, t2, y = reshape_dataset(dataset)
    # use a mds model first
    model = manifold.MDS()
    tf1 = model.fit_transform(t1, y)

    model = manifold.MDS()
    tf2 = model.fit_transform(t2, y)

    return tf1, tf2, y


def sommap_tube(x):
    """Block to process a single tube."""
    x = layers.Conv2D(
        filters=32, kernel_size=3, activation="elu", strides=1,
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="elu", strides=2,
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    # x = layers.MaxPooling2D(pool_size=2, strides=1)(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(filters=64, kernel_size=2, activation="elu", strides=1,
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    x = layers.Conv2D(filters=64, kernel_size=2, activation="elu", strides=2,
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # x = layers.Flatten()(x)
    return x


def sommap_merged(t1, t2):
    """Processing of SOM maps using multiple tubes."""
    t1 = sommap_tube(t1)
    t2 = sommap_tube(t2)
    x = layers.concatenate([t1, t2])

    x = layers.Dense(
        units=256, activation="elu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(GLOBAL_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=128, activation="elu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(GLOBAL_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    return x


def fcs_merged(x):
    """1x1 convolutions on raw FCS data."""
    xa = layers.Conv1D(
        50, 1, strides=1, activation="elu",
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    xa = layers.Conv1D(
        50, 1, strides=1, activation="elu",
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(xa)
    xa = layers.GlobalAveragePooling1D()(xa)
    x = xa
    # xa = layers.BatchNormalization()(xa)

    # xb = layers.Conv1D(16, 1, strides=1, activation="elu")(x)
    # xb = layers.Conv1D(8, 1, strides=1, activation="elu")(x)
    # xb = layers.GlobalMaxPooling1D()(xb)
    # xb = layers.BatchNormalization()(xb)

    # x = layers.concatenate([xa, xb])

    x = layers.Dense(
        50, activation="elu",
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        50, activation="elu",
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    # x = layers.Dense(32, activation="elu")(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(16)(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(16)(x)
    # x = layers.Dropout(0.2)(x)
    return x


def histogram_tube(x):
    """Processing of histogram information using dense neural net."""
    x = layers.Dense(
        units=16, activation="elu", kernel_initializer="uniform",
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    # x = layers.Dropout(rate=0.01)(x)
    x = layers.Dense(
        units=16, activation="elu",
        kernel_regularizer=regularizers.l2(l=GLOBAL_DECAY))(x)
    # x = layers.Dropout(rate=0.01)(x)
    # x = layers.BatchNormalization()(x)
    return x


def histogram_merged(t1, t2):
    """Overall merged processing of histogram information."""
    t1 = histogram_tube(t1)
    t2 = histogram_tube(t2)
    x = layers.concatenate([t1, t2])
    x = layers.Dense(
        units=16, activation="elu",
        kernel_regularizer=regularizers.l2(GLOBAL_DECAY))(x)
    return x


def create_model_convolutional(xshape, yshape):
    """Create a convnet model. The data will be feeded as a 3d matrix."""
    t1 = layers.Input(shape=xshape[0])
    t2 = layers.Input(shape=xshape[1])
    x = sommap_merged(t1, t2)

    final = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=[t1, t2], outputs=final)

    return model


def create_model_histo(xshape, yshape):
    """Create a simple sequential neural network with multiple inputs."""

    t1_input = layers.Input(shape=xshape[0])
    t2_input = layers.Input(shape=xshape[1])

    x = histogram_merged(t1_input, t2_input)
    final = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=[t1_input, t2_input], outputs=final)

    return model


def create_model_fcs(xshape, yshape):
    """Create direct FCS classification model."""
    xinput = layers.Input(shape=xshape[0])

    x = fcs_merged(xinput)
    final = keras.layers.Dense(yshape, activation="softmax")(x)
    model = models.Model(inputs=[xinput], outputs=final)
    return model


def create_model_maphisto(xshape, yshape):
    """Create model using both histogram and SOM map information."""
    m1input = layers.Input(shape=xshape[0])
    m2input = layers.Input(shape=xshape[1])
    t1input = layers.Input(shape=xshape[2])
    t2input = layers.Input(shape=xshape[3])

    mm = sommap_merged(m1input, m2input)
    hm = histogram_merged(t1input, t2input)
    x = layers.concatenate([mm, hm])
    x = layers.Dense(32)(x)
    final = layers.Dense(yshape, activation="softmax")(x)

    model = models.Model(
        inputs=[m1input, m2input, t1input, t2input], outputs=final)
    return model


def create_model_mapfcs(xshape, yshape):
    """Create model combining fcs processing and map cnn."""
    fcsinput = layers.Input(shape=xshape[0])
    m1input = layers.Input(shape=xshape[1])
    m2input = layers.Input(shape=xshape[2])

    fm = fcs_merged(fcsinput)
    mm = sommap_merged(m1input, m2input)
    x = layers.concatenate([fm, mm])
    final = layers.Dense(yshape, activation="softmax")(x)

    model = models.Model(inputs=[fcsinput, m1input, m2input], outputs=final)
    return model


def create_model_all(xshape, yshape):
    """Create model combining fcs, histogram and sommap information."""
    fcsinput = layers.Input(shape=xshape[0])
    m1input = layers.Input(shape=xshape[1])
    m2input = layers.Input(shape=xshape[2])
    t1input = layers.Input(shape=xshape[3])
    t2input = layers.Input(shape=xshape[4])

    fm = fcs_merged(fcsinput)
    mm = sommap_merged(m1input, m2input)
    hm = histogram_merged(t1input, t2input)
    x = layers.concatenate([fm, mm, hm])
    x = layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    final = layers.Dense(yshape, activation="softmax")(x)

    model = models.Model(
        inputs=[fcsinput, m1input, m2input, t1input, t2input], outputs=final)
    return model


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
        model, trainseq, testseq, train_epochs=200, epsilon=1e-8, initial_rate=1e-3, drop=0.5, epochs_drop=100, weights=None, path="mll-sommaps/models", name="0"):
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
            lrate = initial_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
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


def create_metrics_from_pred(pred_df, mapping=None):
    """Create metrics from pred df."""
    groups = [c for c in pred_df.columns if c != "correct"]
    pred = inverse_binarize(pred_df.loc[:, groups].values, classes=groups)
    corr = pred_df["correct"].values

    if mapping is not None:
        map_fun = np.vectorize(lambda n: mapping.get(n, n))
        pred = map_fun(pred)
        corr = map_fun(corr)

        merged_groups = []
        for group in groups:
            mapped = mapping.get(group, group)
            if mapped not in merged_groups:
                merged_groups.append(mapped)
        groups = merged_groups

    stats = {}
    weighted_f1 = metrics.f1_score(corr, pred, average="weighted")
    mcc = metrics.matthews_corrcoef(corr, pred)
    stats["weighted_f1"] = weighted_f1
    stats["mcc"] = mcc
    LOGGER.info("F1: %f", weighted_f1)
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


def main():
    ## CONFIGURATION VARIABLES
    c_uniq_name = "smallernet"
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
    # split train, test using predefined split
    c_predefined_split = True
    c_train_labels = "data/train_labels.json"
    c_test_labels = "data/test_labels.json"
    c_trainargs = {
        "draw_method": "balanced",
        "epoch_size": 16000,
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
        FCSLoader.__name__: {
            "subsample": 100,
        },
        CountLoader.__name__: {
            "version": "dataframe",
        },
        Map2DLoader.__name__: {
            "sel_count": "counts",  # use count in SOM map as just another channel
            "pad_width": 1 if "toroid" in c_sommap_data else 0  # do something more elaborate later
        },
    }
    ## END CONFIGURATION VARIABLES

    # save configuration variables
    config_dict = locals()

    name = f"{c_uniq_name}_{c_model}_{c_groupmap}"

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

    group_maps = {
        "8class": {
            "groups": ["CM", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"],
            "map": {"CLL": "CM", "MBL": "CM"}
        },
        "6class": {
            "groups": ["CM", "MP", "LM", "FL", "HCL", "normal"],
            "map": {
                "CLL": "CM",
                "MBL": "CM",
                "MZL": "LM",
                "LPL": "LM",
                "MCL": "MP",
                "PL": "MP",
            }
        },
        "5class": {
            "groups": ["CM", "MP", "LMF", "HCL", "normal"],
            "map": {
                "CLL": "CM",
                "MBL": "CM",
                "MZL": "LMF",
                "LPL": "LMF",
                "FL": "LMF",
                "LM": "LMF",
                "MCL": "MP",
                "PL": "MP",
            }
        }
    }
    groups = group_maps[c_groupmap]["groups"]
    group_map = group_maps[c_groupmap]["map"]

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
        ## simpler group weights
        group_weights = {
            ("normal", None): (1.0, 20.0),
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

    if c_model == "histogram":
        xoutputs = [
            CountLoader.create_inferred(train, tube=1, **c_dataoptions[CountLoader.__name__]),
            CountLoader.create_inferred(train, tube=2, **c_dataoptions[CountLoader.__name__]),
        ]
        train_batch = 64
        test_batch = 128
        modelfun = create_model_histo
    elif c_model == "sommap":
        xoutputs = [
            Map2DLoader.create_inferred(train, tube=1, **c_dataoptions[Map2DLoader.__name__]),
            Map2DLoader.create_inferred(train, tube=2, **c_dataoptions[Map2DLoader.__name__]),
        ]
        train_batch = 16
        test_batch = 32
        modelfun = create_model_convolutional
    elif c_model == "maphisto":
        xoutputs = [
            Map2DLoader.create_inferred(train, tube=1, **c_dataoptions[Map2DLoader.__name__]),
            Map2DLoader.create_inferred(train, tube=2, **c_dataoptions[Map2DLoader.__name__]),
            CountLoader.create_inferred(train, tube=1, **c_dataoptions[CountLoader.__name__]),
            CountLoader.create_inferred(train, tube=2, **c_dataoptions[CountLoader.__name__]),
        ]
        train_batch = 32
        test_batch = 32
        modelfun = create_model_maphisto
    elif c_model == "etefcs":
        xoutputs = [
            FCSLoader.create_inferred(train, tubes=[1, 2], **c_dataoptions[FCSLoader.__name__]),
        ]
        train_batch = 16
        test_batch = 16
        modelfun = create_model_fcs
    elif c_model == "mapfcs":
        xoutputs = [
            FCSLoader.create_inferred(train, tubes=[1, 2], **c_dataoptions[FCSLoader.__name__]),
            Map2DLoader.create_inferred(train, tube=1, **c_dataoptions[Map2DLoader.__name__]),
            Map2DLoader.create_inferred(train, tube=2, **c_dataoptions[Map2DLoader.__name__]),
        ]
        train_batch = 32
        test_batch = 32
        modelfun = create_model_mapfcs
    elif c_model == "maphistofcs":
        xoutputs = [
            FCSLoader.create_inferred(train, tubes=[1, 2], **c_dataoptions[FCSLoader.__name__]),
            Map2DLoader.create_inferred(train, tube=1, **c_dataoptions[Map2DLoader.__name__]),
            Map2DLoader.create_inferred(train, tube=2, **c_dataoptions[Map2DLoader.__name__]),
            CountLoader.create_inferred(train, tube=1, **c_dataoptions[CountLoader.__name__]),
            CountLoader.create_inferred(train, tube=2, **c_dataoptions[CountLoader.__name__]),
        ]
        train_batch = 32
        test_batch = 32
        modelfun = create_model_all
    else:
        raise RuntimeError(f"Unknown model {c_model}")

    modelpath = pathlib.Path(f"{c_output_model}/{name}")
    trainseq = SOMMapDataset(train, xoutputs, batch_size=train_batch, groups=groups, **c_trainargs)
    testseq = SOMMapDataset(test, xoutputs, batch_size=test_batch, groups=groups, **c_testargs)
    model = modelfun(*trainseq.shape)
    pred_df = run_save_model(
        model, trainseq, testseq, weights=weights, path=modelpath, name="0", **c_runargs)

    LOGGER.info(f"Statistics results for {name}")
    for gname, groupstat in group_maps.items():
        # skip if our cohorts are larger
        if len(groups) < len(groupstat["groups"]):
            continue

        LOGGER.info(f"-- {len(groupstat['groups'])} --")
        conf, stats = create_metrics_from_pred(pred_df, mapping=groupstat["map"])
        plotting.plot_confusion_matrix(
            conf.values, groupstat["groups"], normalize=True,
            title=f"Confusion matrix (f1 {stats['weighted_f1']:.2f} mcc {stats['mcc']:.2f})",
            filename=outpath / f"confusion_{gname}", dendroname=None)
        conf.to_csv(outpath / f"confusion_{gname}.csv")
        with open(str(outpath / f"stats_{gname}.json"), "w") as jsfile:
            json.dump(stats, jsfile)


if __name__ == "__main__":
    main()
