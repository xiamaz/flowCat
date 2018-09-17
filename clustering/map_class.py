import os
import pickle
import pathlib
import functools
import hashlib

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
            }
        except KeyError as e:
            print(f"{e} - Not found in histo or sommap")
            continue
        except AssertionError as e:
            print(f"{case.id}|{case.group} - {e}")
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
                print(missing)
                raise RuntimeError()
            counts = sel_rows.values
        return counts


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
                preprocessing.MinMaxScaler().fit_transform(data),
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
    def __init__(self, tube, gridsize, channels, sel_count=None, pad_width=0):
        """Object to transform input rows into the specified format."""
        self.tube = tube
        self.gridsize = gridsize
        self.channels = channels
        self.sel_count = sel_count
        self.pad_width = pad_width

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

    def __call__(self, data):
        """Output specified format."""
        map_list = []
        for path in data[self.datacol]:
            mapdata = self.load_data(path, self.tube)
            mapdata = select_drop_counts(mapdata, self.sel_count)
            data = reshape_dataframe(
                mapdata,
                m=self.gridsize,
                n=self.gridsize,
                pad_width=self.pad_width)
            map_list.append(data)

        return np.stack(map_list)


class SOMMapDataset(LoaderMixin, keras.utils.Sequence):
    """Dataset for creating and yielding batches of data for keras model.

    Data can be generated by random draw or alternatively in sequence.
    """

    def __init__(
            self, data, xoutputs,
            batch_size=32, draw_method="shuffle", epoch_size=None,
            groups=None, toroidal=False
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
                ]
            epoch_size: Number of samples in a single epoch. Is data length if None or 0.
            groups: List of groups to transform the labels into binary matrix.
            toroidal: Pad data in toroidal manner.
        Returns:
            SOMMapDataset object.
        """
        self.batch_size = batch_size
        self.draw_method = draw_method

        if self.draw_method == "shuffle":
            self._data = data.sample(frac=1)
        elif self.draw_method == "sequential":
            self._data = data
        elif self.draw_method == "balanced":
            sample_num = epoch_size or len(data)
            self._data = data.groupby("group").apply(lambda x: x.sample(
                n=sample_num, replace=True)).reset_index(0, drop=True).sample(frac=1)
        else:
            raise RuntimeError(
                f"Unknown draw method: {self.draw_method}. "
                "Valid options are: ['shuffle', 'sequential', 'balanced']")

        self.epoch_size = epoch_size if epoch_size else len(self._data)

        if groups is None:
            groups = list(self._data["group"].unique())

        self.groups = groups

        if toroidal:
            pad_width = 1
        else:
            pad_width = 0
        self._xoutputs = xoutputs

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
    x = layers.Conv2D(filters=32, kernel_size=2, activation="relu", strides=1)(x)
    x = layers.Conv2D(filters=64, kernel_size=2, activation="relu", strides=2)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(filters=64, kernel_size=1, activation="relu", strides=1)(x)
    x = layers.Conv2D(filters=64, kernel_size=2, activation="relu", strides=1)(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    return x


def sommap_merged(t1, t2):
    """Processing of SOM maps using multiple tubes."""
    t1 = sommap_tube(t1)
    t2 = sommap_tube(t2)
    x = layers.concatenate([t1, t2])

    x = layers.Dense(
        units=64, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=64, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.2)(x)
    return x


def fcs_merged(x):
    """1x1 convolutions on raw FCS data."""
    xa = layers.Conv1D(64, 1, strides=1, activation="relu")(x)
    xa = layers.GlobalAveragePooling1D()(xa)
    # xa = layers.BatchNormalization()(xa)

    xb = layers.Conv1D(16, 1, strides=1, activation="relu")(x)
    xb = layers.GlobalMaxPooling1D()(xb)
    xb = layers.BatchNormalization()(xb)

    x = layers.concatenate([xa, xb])

    x = layers.Dense(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16)(x)
    x = layers.Dropout(0.2)(x)
    return x


def histogram_tube(x):
    """Processing of histogram information using dense neural net."""
    x = layers.Dense(
        units=32, activation="relu", kernel_initializer="uniform")(x)
    x = layers.Dropout(rate=0.01)(x)
    x = layers.Dense(
        units=32, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l1(.01))(x)
    x = layers.Dropout(rate=0.01)(x)
    x = layers.BatchNormalization()(x)
    return x


def histogram_merged(t1, t2):
    """Overall merged processing of histogram information."""
    t1 = histogram_tube(t1)
    t2 = histogram_tube(t2)
    x = layers.concatenate([t1, t2])
    x = layers.Dense(
        units=16, activation="relu", kernel_initializer="uniform")(x)
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
    x = layers.Dense(32)(x)
    final = layers.Dense(yshape, activation="softmax")(x)

    model = models.Model(
        inputs=[fcsinput, m1input, m2input, t1input, t2input], outputs=final)
    return model


def classify_histogram(train, test, groups=None, weights=None, *args, **kwargs):
    """Extremely simple sequential neural network with two
    inputs for the 10x10x12 data
    """
    xoutputs = [
        CountLoader.create_inferred(
            train, tube=1, version="dataframe"),
        CountLoader.create_inferred(
            train, tube=2, version="dataframe"),
    ]

    trainseq = SOMMapDataset(train, xoutputs, batch_size=64, draw_method="balanced", groups=groups, epoch_size=8000)
    testseq = SOMMapDataset(test, xoutputs, batch_size=128, draw_method="sequential", groups=groups)

    model = create_model_histo(*trainseq.shape)

    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights)
    model.compile(
        loss=lossfun,
        optimizer="adam",
        metrics=["acc"]
    )
    return run_save_model(model, trainseq, testseq, *args, **kwargs)


def classify_convolutional(
        train, test, weights=None, toroidal=False, groups=None, *args, **kwargs
):
    # wrap pad input matrix if we use toroidal input data
    pad_width = 1 if toroidal else 0

    xoutputs = [
        Map2DLoader.create_inferred(
            train, tube=1, pad_width=pad_width, sel_count=None),
        Map2DLoader.create_inferred(
            train, tube=2, pad_width=pad_width, sel_count=None),
    ]
    trainseq = SOMMapDataset(train, xoutputs, batch_size=64, draw_method="balanced", groups=groups, epoch_size=8000)
    testseq = SOMMapDataset(test, xoutputs, batch_size=128, draw_method="sequential", groups=groups)

    model = create_model_convolutional(*trainseq.shape)

    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights)
    model.compile(
        loss=lossfun,
        optimizer=optimizers.Adam(
            lr=0.0001, decay=0.0, epsilon=0.0001
        ),
        metrics=["acc"]
    )
    return run_save_model(model, trainseq, testseq, *args, **kwargs)


def classify_fcs(train, test, weights=None, groups=None, *args, **kwargs):
    xoutputs = [
        FCSLoader.create_inferred(train, tubes=[1, 2], subsample=500),
    ]
    trainseq = SOMMapDataset(
        train, xoutputs, batch_size=16, draw_method="balanced", groups=groups, epoch_size=8000)
    testseq = SOMMapDataset(
        test, xoutputs, batch_size=16, draw_method="sequential", groups=groups)

    model = create_model_fcs(*trainseq.shape)

    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights)
    model.compile(
        loss=lossfun,
        optimizer=optimizers.Adam(
            lr=0.0001, decay=0.0, epsilon=0.1
        ),
        metrics=["acc"]
    )
    return run_save_model(model, trainseq, testseq, *args, **kwargs)


def classify_mapfcs(
        train, test, weights=None, toroidal=False, groups=None, *args, **kwargs
):
    pad_width = 1 if toroidal else 0
    xoutputs = [
        FCSLoader.create_inferred(train, tubes=[1, 2], subsample=200),
        Map2DLoader.create_inferred(
            train, tube=1, pad_width=pad_width, sel_count=None),
        Map2DLoader.create_inferred(
            train, tube=2, pad_width=pad_width, sel_count=None),
    ]
    trainseq = SOMMapDataset(
        train, xoutputs, batch_size=32, draw_method="balanced", groups=groups,
        epoch_size=8000)
    testseq = SOMMapDataset(
        test, xoutputs, batch_size=32, draw_method="sequential", groups=groups)

    model = create_model_mapfcs(*trainseq.shape)

    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights)
    model.compile(
        loss=lossfun,
        optimizer=optimizers.Adam(
            lr=0.0001, decay=0.0, epsilon=0.001
        ),
        metrics=["acc"]
    )
    return run_save_model(model, trainseq, testseq, *args, **kwargs)


def classify_all(
        train, test, weights=None, toroidal=False, groups=None, *args, **kwargs
):
    pad_width = 1 if toroidal else 0
    xoutputs = [
        FCSLoader.create_inferred(train, tubes=[1, 2], subsample=200),
        Map2DLoader.create_inferred(
            train, tube=1, pad_width=pad_width, sel_count=None),
        Map2DLoader.create_inferred(
            train, tube=2, pad_width=pad_width, sel_count=None),
        CountLoader.create_inferred(
            train, tube=1, version="dataframe"),
        CountLoader.create_inferred(
            train, tube=2, version="dataframe"),
    ]
    trainseq = SOMMapDataset(
        train, xoutputs, batch_size=32, draw_method="balanced", groups=groups,
        epoch_size=8000)
    testseq = SOMMapDataset(
        test, xoutputs, batch_size=32, draw_method="sequential", groups=groups)

    model = create_model_all(*trainseq.shape)

    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights)
    model.compile(
        loss=lossfun,
        optimizer=optimizers.Adam(
            lr=0.0001, decay=0.0, epsilon=0.1
        ),
        metrics=["acc"]
    )
    return run_save_model(model, trainseq, testseq, *args, **kwargs)


def run_save_model(model, trainseq, testseq, path="mll-sommaps/models", name="0"):
    """Run and predict using the given model. Also save the model in the given
    path with specified name."""
    history = model.fit_generator(
        trainseq, epochs=30,
        callbacks=[
            # keras.callbacks.EarlyStopping(min_delta=0.01, patience=20, mode="min")
        ],
        validation_data=testseq,
        # class_weight={
        #     0: 1.0,  # CM
        #     1: 2.0,  # MCL
        #     2: 2.0,  # PL
        #     3: 2.0,  # LPL
        #     4: 2.0,  # MZL
        #     5: 50.0,  # FL
        #     6: 50.0,  # HCL
        #     7: 1.0,  # normal
        # }
        workers=4,
        use_multiprocessing=True,
    )
    pred_mat = model.predict_generator(testseq, workers=4, use_multiprocessing=True)

    # save the model weights after training
    modelpath = pathlib.Path(path)
    modelpath.mkdir(parents=True, exist_ok=True)
    model.save(modelpath / f"model_{name}.h5")
    with open(str(modelpath / f"history_{name}.p"), "wb") as hfile:
        pickle.dump(history.history, hfile)

    pred_df = pd.DataFrame(
        pred_mat, columns=trainseq.groups, index=testseq.labels)
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
    print("F1: ", weighted_f1)
    print("MCC: ", mcc)

    confusion = metrics.confusion_matrix(corr, pred, groups,)
    confusion = pd.DataFrame(confusion, columns=groups, index=groups)
    print("Confusion matrix")
    print(confusion)
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
    return weights


def split_data(data, test_num=0.2):
    """Split data in stratified fashion by group.
    Args:
        data: Dataset to be split. Label should be contained in 'group' column.
        test_num: Ratio of samples in test per group or absolute number of samples in each group for test.
    Returns:
        (train, test) with same columns as input data.
    """
    grouped = data.groupby("group")
    if test_num < 1:
        test = grouped.apply(lambda d: d.sample(frac=test_num)).reset_index(level=0, drop=True)
    else:
        group_sizes = grouped.size()
        if any(group_sizes <= test_num):
            insuff = group_sizes[group_sizes <= test_num]
            print("Insufficient sizes: ", insuff)
            raise RuntimeError("Some cohorts are too small.")
        test = grouped.apply(lambda d: d.sample(n=test_num)).reset_index(level=0, drop=True)
    train = data.drop(test.index, axis=0)
    return train, test


def main():
    indata = load_dataset(
        "mll-sommaps/sample_maps/selected1_toroid_s32",
        histopath="../mll-flow-classification/clustering/abstract/abstract_somgated_1_20180723_1217",
        fcspath="s3://mll-flowdata/CLL-9F"
    )
    # # save the data
    # with open("indata_selected5_somgated_fcs.p", "wb") as f:
    #     pickle.dump(indata, f)
    # return

    # # load the data again
    # with open("indata_selected5_somgated_fcs.p", "rb") as f:
    #     indata = pickle.load(f)

    # groups = ["CLL", "MBL", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"]
    # 8-class
    groups = ["CM", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"]
    group_map = {
        "CLL": "CM",
        "MBL": "CM",
    }

    # # 6-class
    groups6 = ["CM", "MP", "LM", "FL", "HCL", "normal"]
    g6_map = {
        "CLL": "CM",
        "MBL": "CM",
        "MZL": "LM",
        "LPL": "LM",
        "MCL": "MP",
        "PL": "MP",
    }
    # 5-class
    groups5 = ["CM", "MP", "LMF", "HCL", "normal"]
    g5_map = {
        "CLL": "CM",
        "MBL": "CM",
        "MZL": "LMF",
        "LPL": "LMF",
        "FL": "LMF",
        "MCL": "MP",
        "PL": "MP",
    }

    indata["orig_group"] = indata["group"]
    indata = modify_groups(indata, mapping=group_map)
    indata = indata.loc[indata["group"].isin(groups), :]

    # Group weights are a dict mapping tuples to tuples. Weights are for
    # false classifications in the given direction.
    # (a, b) --> (a>b, b>a)
    group_weights = {
        ("normal", None): (10.0, 100.0),
        ("MZL", "LPL"): (2, 2),
        ("MCL", "PL"): (2, 2),
        ("FL", "LPL"): (3, 5),
        ("FL", "MZL"): (3, 5),
    }
    weights = create_weight_matrix(group_weights, groups, base_weight=5)
    weights = None

    # plotpath = pathlib.Path("sommaps/output/lotta")
    # tf1, tf2, y = decomposition(indata)
    # plot_transformed(plotpath, tf1, tf2, y)
    validation = "holdout"
    name = "convolutional"

    train, test = split_data(indata, test_num=0.2)

    # pred_dfs = classify_histogram(
    #     train, test, weights=weights, groups=groups, path=f"mll-sommaps/models/{name}",
    # )

    pred_df = classify_convolutional(
        train, test, toroidal=True, weights=weights,
        groups=groups, path=f"mll-sommaps/models/{name}")

    # pred_dfs = classify_fcs(
    #     train, test, groups=groups, path=f"mll-sommaps/models/{name}")

    # pred_dfs = classify_mapfcs(
    #     train, test, toroidal=True, weights=weights,
    #     groups=groups, path=f"mll-sommaps/models/{name}")

    # pred_dfs = classify_all(
    #     train, test, toroidal=True, weights=weights,
    #     groups=groups, path=f"mll-sommaps/models/{name}")

    outpath = pathlib.Path(f"output/{name}_{validation}")
    outpath.mkdir(parents=True, exist_ok=True)

    # normal f(8)
    conf_8, stats_8 = create_metrics_from_pred(pred_df, mapping=None)
    plotting.plot_confusion_matrix(
        conf_8.values, groups, normalize=True,
        filename=outpath / "confusion_8class", dendroname=None)
    # merged f(6)
    conf_6, stats_6 = create_metrics_from_pred(pred_df, mapping=g6_map)
    plotting.plot_confusion_matrix(
        conf_6.values, groups6, normalize=True,
        filename=outpath / "confusion_6class", dendroname=None)
    # merged f(5)
    conf_5, stats_5 = create_metrics_from_pred(pred_df, mapping=g5_map)
    plotting.plot_confusion_matrix(
        conf_5.values, groups5, normalize=True,
        filename=outpath / "confusion_5class", dendroname=None)


if __name__ == "__main__":
    main()
