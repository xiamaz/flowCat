import pickle
import pathlib
import functools

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


COLS = "grcmyk"

def get_tubepaths(label, cases):
    matched = None
    for case in cases:
        if case.id == label:
            matched = case
            break
    tubepaths = {k: v[-1].path for k, v in matched.tubepaths.items()}
    return tubepaths


def load_histolabels(histopath):
    dfs = [set(pd.read_csv(f"{histopath}/tube{t}.csv", index_col=0)["label"]) for t in [1, 2]]
    both_labels = functools.reduce(lambda x, y: x & y, dfs)
    return both_labels

def load_dataset(mappath, histopath, fcspath):
    """Return dataframe containing columns with filename and labels."""
    mappath = pathlib.Path(mappath)

    sommap_labels = set(pd.read_csv(f"{mappath}.csv", index_col=0)["label"])
    histo_labels = load_histolabels(histopath)
    both_labels = sommap_labels & histo_labels

    assert both_labels, "No data having both histo and sommap info."

    cdict = {}
    cases = cc.CaseCollection(fcspath, tubes=[1, 2])
    for case in cases:
        if case.id not in both_labels:
            continue
        cdict[case.id] = {
            "group": case.group,
            "sommappath": str(mappath / f"{case.id}_t{{tube}}.csv"),
            "fcspath": {k: v[-1].path for k, v in case.tubepaths.items()},
            "histopath": f"{histopath}/tube{{tube}}.csv",
        }

    dataset = pd.DataFrame.from_dict(cdict, orient="index")
    return dataset


def subtract_ref_data(data, references):
    data["data"] = data["data"].apply(
        lambda t: [r - a for r, a in zip(references, t)]
    )
    return data


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
    datacol = "sommap_path"
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
        non_number_cols = [c for c in dfdata if not c.isdigit()]
        dfdata.set_index(dfdata["label"], inplace=True)
        # dfdata.drop(non_number_cols, inplace=True, axis=1)
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
            sel_rows = self.data.loc[data["label"], :]
            counts = sel_rows.values
            print(data.shape, sel_rows.shape)
            print(sel_rows)
        return counts


class FCSLoader(LoaderMixin):
    """Directly load FCS data associated with the given ids."""
    def __init__(self, tubes, channels=None, subsample=200):
        self.tubes = tubes
        self.channels = channels
        self.subsample = subsample

    @classmethod
    def create_inferred(cls, data, tubes, subsample=200, channels=None, *args, **kwargs):
        testdata = cls._load_data(
            data[cls.fcscol].iloc[0], subsample, tubes=tubes, channels=channels
        )
        channels = list(testdata.columns)
        return cls(tubes=tubes, channels=channels, subsample=subsample, *args, **kwargs)

    @property
    def shape(self):
        return (self.subsample, len(self.channels))

    @staticmethod
    def _load_data(pathdict, subsample, tubes, channels):
        datas = []
        for tube in tubes:
            _, data = fcsparser.parse(pathdict[tube], data_set=0, encoding="latin-1")

            data.drop([c for c in data.columns if "nix" in c], axis=1, inplace=True)

            data = pd.DataFrame(
                preprocessing.StandardScaler().fit_transform(data),
                columns=data.columns)

            data = data.sample(n=subsample)

            cols = [c+s for c in data.columns for s in ["", "sig"]]
            sig_cols = [c for c in cols if c.endswith("sig")]
            data = pd.concat(
                [data, pd.DataFrame(1, columns=sig_cols, index=data.index)], axis=1)
            data = data.loc[:, cols]
            datas.append(data)

        merged = pd.concat(datas, sort=False)
        return merged.fillna(0)

    def __call__(self, data):
        mapped_fcs = []
        for path in data[self.fcscol]:
            mapped_fcs.append(self._load_data(
                path, self.subsample, self.tubes, self.channels
            )[self.channels].values)
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
            self.gridsize, self.gridsize, len(self.channels) + bool(self.sel_count)
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



    """Dataset for creating and yielding batches of data for keras model.

    Data can be generated by random draw or alternatively in sequence.
    """

    def __init__(
            self, data,
            batch_size=32, draw_method="shuffle", epoch_size=None,
            groups=None, toroidal=False,
    ):
        """
        Args:
            data: DataFrame containing labels and paths to data.
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
                n=sample_num, replace=True)).reset_index(drop=True).sample(frac=1)
        else:
            raise RuntimeError(
                f"Unknown draw method: {self.draw_method}. "
                "Valid options are: ['shuffle', 'sequential', 'balanced']")

        self.epoch_size = epoch_size if epoch_size else len(self._data)

        if groups is None:
            groups = list(self._data["group"].unique())

        self.binarizer = preprocessing.LabelBinarizer()
        self.binarizer.fit(groups)

        if toroidal:
            pad_width = 1
        else:
            pad_width = 0

        self._xoutputs = [
            Map2DLoader.create_inferred(
                self._data, tube=1, pad_width=pad_width, sel_count="counts"),
            Map2DLoader.create_inferred(
                self._data, tube=2, pad_width=pad_width, sel_count="counts"),
            CountLoader.create_inferred(
                self._data, tube=1, version="dataframe"),
            CountLoader.create_inferred(
                self._data, tube=2, version="dataframe"),
            FCSLoader.create_inferred(
                self._data, tubes=[1, 2], subsample=200),
        ]

    @property
    def xshape(self):
        """Return shape of xvalues. Should be a list of shapes describing each input.
        """
        return [x.shape for x in self._xoutputs]

    @property
    def yshape(self):
        """Return shape of yvalues."""
        return len(self.binarizer.classes_)

    @property
    def shape(self):
        """Return tuple of xshape and yshape."""
        return self.xshape, self.yshape

    def __len__(self):
        """Return the number of batches generated."""
        return int(np.ceil(self.epoch_size / float(self.batch_size)))

    def __getitem__(self, idx):
        """Get a single batch by id."""
        batch_data = self._data.iloc[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        xdata = [x(batch_data) for x in self._xoutputs]

        ydata = batch_data["group"]
        ybinary = self.binarizer.transform(ydata)
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


def identity_block(input_tensor, kernel_size, filters, stage, block, tnum=0):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = f'{tnum}_res' + str(stage) + block + '_branch'
    bn_name_base = f'{tnum}_bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               tnum=0,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = f'{tnum}_res' + str(stage) + block + '_branch'
    bn_name_base = f'{tnum}_bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def create_resnet(xshape, yshape, num_inputs=2, classweights=None):
    """Create resnet."""
    inputs = []
    input_ends = []

    for num in range(num_inputs):
        i = layers.Input(shape=xshape)
        inputs.append(i)
        x = i

        # x = layers.Conv2D(
        #     64, (5, 5),
        #     strides=(2, 2), padding="valid", kernel_initializer="he_normal",
        #     name="conv1")(x)
        # x = layers.BatchNormalization(axis=3)(x)
        # x = layers.Activation("relu")(x)
        # x = layers.ZeroPadding2D(padding=(1, 1))(x)
        # x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1), tnum=num)
        x = identity_block(x, 3, [32, 32, 128], stage=2, block='b', tnum=num)
        x = identity_block(x, 3, [32, 32, 128], stage=2, block='c', tnum=num)

        x = conv_block(x, 3, [64, 64, 256], stage=3, block='a', tnum=num)
        x = identity_block(x, 3, [64, 64, 256], stage=3, block='b', tnum=num)
        x = identity_block(x, 3, [64, 64, 256], stage=3, block='c', tnum=num)
        x = identity_block(x, 3, [64, 64, 256], stage=3, block='d', tnum=num)

        x = conv_block(x, 3, [128, 128, 512], stage=4, block='a', tnum=num)
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='b', tnum=num)
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='c', tnum=num)
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='d', tnum=num)
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='e', tnum=num)
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='f', tnum=num)

        x = conv_block(x, 3, [256, 256, 1024], stage=5, block='a', tnum=num)
        x = identity_block(x, 3, [256, 256, 1024], stage=5, block='b', tnum=num)
        x = identity_block(x, 3, [256, 256, 1024], stage=5, block='c', tnum=num)

        x = layers.GlobalAveragePooling2D(name=f'{num}_avg_pool')(x)
        input_ends.append(x)

    x = layers.concatenate(input_ends)
    x = layers.Dense(yshape, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)

    lossfun = "categorical_crossentropy"
    model.compile(
        loss=lossfun,
        optimizer=optimizers.Adam(
            lr=0.0001, decay=0.0, epsilon=0.1
        ),
        metrics=["acc"]
    )
    return model


def create_model_convolutional(
        xshape, yshape, num_inputs=2
):
    """Create a convnet model. The data will be feeded as a 3d matrix."""
    inputs = []
    input_ends = []
    for i in range(num_inputs):
        t_input = layers.Input(shape=xshape)

        x = t_input
        x = layers.Conv2D(filters=32, kernel_size=2, activation="relu", strides=1)(x)
        x = layers.Conv2D(filters=64, kernel_size=2, activation="relu", strides=2)(x)
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(filters=64, kernel_size=2, activation="relu", strides=1)(x)
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # x = layers.Flatten()(x)

        input_ends.append(x)
        inputs.append(t_input)

    x = layers.concatenate(input_ends)

    x = layers.Dense(
        units=256, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=128, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    final = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=inputs, outputs=final)

    return model


def create_model(xshape, yshape, num_inputs=2, classweights=None):
    """Create a simple sequential neural network with multiple inputs."""

    inputs = []
    input_ends = []
    for i in range(num_inputs):
        t_input = layers.Input(shape=xshape)

        # t_attention = layers.Dense(
        #     units=xshape[0], activation="softmax",
        #     kernel_regularizer=regularizers.l2(.01)
        # )(t_input)
        # t_multatt = layers.multiply([t_input, t_attention])

        t_a = layers.Dense(
            units=128, activation="relu", kernel_initializer="uniform",
            # kernel_regularizer=regularizers.l1(.01)
        )(t_input)
        t_ad = layers.Dropout(rate=0.01)(t_a)

        t_b = layers.Dense(
            units=64, activation="relu", kernel_initializer="uniform",
            kernel_regularizer=regularizers.l1(.01)
        )(t_ad)
        t_bd = layers.Dropout(rate=0.01)(t_b)

        t_end = layers.BatchNormalization(
        )(t_b)

        input_ends.append(t_end)
        inputs.append(t_input)

    concat = layers.concatenate(input_ends)
    m_end = layers.Dense(
        units=64, activation="relu", kernel_initializer="uniform"
    )(concat)

    final = layers.Dense(
        units=yshape, activation="softmax"
    )(m_end)

    model = models.Model(inputs=inputs, outputs=final)

    if classweights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=classweights)

    model.compile(
        loss=lossfun,
        optimizer="adam",
        metrics=["acc"]
    )

    return model


def classify(data):
    """Extremely simple sequential neural network with two
    inputs for the 10x10x12 data
    """
    groups = list(data["group"].unique())

    train, test = model_selection.train_test_split(
        data, train_size=0.8, stratify=data["group"])

    train = pd.concat([train, test])

    tr1, tr2, ytrain = reshape_dataset(train)

    binarizer = preprocessing.LabelBinarizer()
    binarizer.fit(groups)
    ytrain_mat = binarizer.transform(ytrain)

    te1, te2, ytest = reshape_dataset(test)
    ytest_mat = binarizer.transform(ytest)

    model = naive_bayes.GaussianNB()
    model.fit(tr1, ytrain)

    model = create_model((tr1.shape[1], ), ytrain_mat.shape[1], classweights=None)
    model.fit([tr1, tr2], ytrain_mat, epochs=20, batch_size=16, validation_split=0.2)
    pred = model.predict([te1, te2], batch_size=128)
    pred = binarizer.inverse_transform(pred)

    res = model.score(te1, ytest)
    print(res)
    pred = model.predict(te1)

    print("F1: ", metrics.f1_score(ytest, pred, average="micro"))

    confusion = metrics.confusion_matrix(ytest, pred, binarizer.classes_,)
    stats = {"mcc": metrics.matthews_corrcoef(ytest, pred)}
    return stats, confusion, binarizer.classes_


def classify_convolutional(
        train, test, m=10, n=10, weights=None, toroidal=False, groups=None,
        path="mll-sommaps/models", name="0"
):
    if groups is None:
        groups = list(data["group"].unique())

    trainseq = SOMMapDataset(train, batch_size=64, draw_method="shuffle", groups=groups)
    testseq = SOMMapDataset(test, batch_size=128, draw_method="sequential", groups=groups)

    model = create_model_convolutional(trainseq.xshape, trainseq.yshape)
    if weights is None:
        lossfun = "categorical_crossentropy"
    else:
        lossfun = weighted_crossentropy.WeightedCategoricalCrossEntropy(
            weights=weights)

    model.compile(
        loss=lossfun,
        optimizer=optimizers.Adam(
            lr=0.0001, decay=0.0, epsilon=0.00001
        ),
        metrics=["acc"]
    )
    # model = create_resnet(tr1[0].shape, len(groups), classweights=weights)
    history = model.fit_generator(
        trainseq, epochs=100,
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
    )
    pred_mat = model.predict_generator(testseq, workers=4)

    # save the model weights after training
    modelpath = pathlib.Path(path)
    modelpath.mkdir(parents=True, exist_ok=True)
    model.save(modelpath / f"model_{name}.h5")
    with open(str(modelpath / f"history_{name}.p"), "wb") as hfile:
        pickle.dump(history.history, hfile)

    pred_df = pd.DataFrame(
        pred_mat, columns=groups, index=data.loc[test_index, "label"])
    pred_df["correct"] = ytest.tolist()
    pred_df.to_csv(modelpath / f"predictions_{name}.csv")
    create_metrics_from_pred(pred_df)
    return pred_df


def create_metrics_from_pred(pred_df, mapping=None):
    """Create metrics from pred df."""
    lb = preprocessing.LabelBinarizer()
    groups = [c for c in pred_df.columns if c != "correct"]
    lb.fit(groups)
    pred = lb.inverse_transform(pred_df.loc[:, groups].values)
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


def main():
    map_size = 32

    indata = load_dataset(
        "mll-sommaps/sample_maps/selected5_toroid_s32",
        histopath="../mll-flow-classification/clustering/abstract/abstract_somgated_1_20180723_1217",
        fcspath="/home/zhao/tmp/CLL-9F"
    )
    # save the data
    with open("indata_selected5_somgated_fcs.p", "wb") as f:
        pickle.dump(indata, f)
    return

    # load the data again
    with open("indata_selected5_somgated_fcs.p", "rb") as f:
        indata = pickle.load(f)

    # TODO there are duplicates in the upsampling data
    thist = pd.read_csv(
        "../mll-flow-classification/clustering/abstract/abstract_somgated_1_20180723_1217/tube1.csv",
        index_col=0)
    thist.loc[thist.duplicated("label", keep=False), :].sort_values("label")

    with open("/home/zhao/tmp/CLL-9F/case_info.json") as f:
        cinfo = json.load(f)



    test = SOMMapDataset(indata, draw_method="shuffle", epoch_size=1000)
    # groups = ["CLL", "MBL", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"]
    # 8-class
    groups = ["CM", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"]
    group_map = {
        "CLL": "CM",
        "MBL": "CM",
    }

    # # 6-class
    merged_groups = ["CM", "MP", "LM", "FL", "HCL", "normal"]
    merged_group_map = {
        "CLL": "CM",
        "MBL": "CM",
        "MZL": "LM",
        "LPL": "LM",
        "MCL": "MP",
        "PL": "MP",
    }
    indata = modify_groups(indata, mapping=group_map)
    indata = indata.loc[indata["group"].isin(groups), :]
    indata.reset_index(drop=True, inplace=True)
    # Group weights are a dict mapping tuples to tuples. Weights are for
    # false classifications in the given direction.
    # (a, b) --> (a>b, b>a)
    group_weights = {
        ("normal", None): (10.0, 100.0),
        ("MBL", "CLL"): (2, 2),
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
    name = "selected5_toroid_8class_60test_ep100"

    train, test = split_data(indata, num=60)

    # n_metrics, n_confusion, n_groups = classify(normdata)
    pred_dfs = classify_convolutional(
        train, test, m=map_size, n=map_size, toroidal=False, weights=weights,
        groups=groups, path=f"mll-sommaps/models/{name}")

    # pred_df_8class = pd.read_csv(
    #     "mll-sommaps/models/cllall1_planar_8class_60test_ep100/predictions_0.csv",
    #     index_col=0)
    # n6_merged = create_metrics_from_pred(pred_df_8class, merged_group_map)
    # pred_df_6class = pd.read_csv(
    #     "mll-sommaps/models/cllall1_planar_60test_ep100/predictions_0.csv",
    #     index_col=0)
    # n6_direct = create_metrics_from_pred(pred_df_6class)

    outpath = pathlib.Path(f"output/{name}_{validation}")
    outpath.mkdir(parents=True, exist_ok=True)

    plotting.plot_confusion_matrix(
        sum_confusion, groups, normalize=True,
        filename=outpath / "confusion_merged_weighted", dendroname=outpath / "dendro_merged_weighted"
    )


if __name__ == "__main__":
    main()
