import pathlib

import numpy as np
import pandas as pd
from sklearn import manifold, model_selection, preprocessing
from sklearn import naive_bayes
from sklearn import metrics

from keras import layers, models, regularizers

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import sys
sys.path.append("../classification")
from classification import plotting


COLS = "grcmyk"


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


def load_dataset(path):
    """Return dataframe containing columns with filename and labels."""

    labels = pd.read_csv(f"{path}.csv", index_col=0)

    labels.columns = ["label", "group"]

    labels["data"] = labels["label"].apply(
        lambda i: [
            pd.read_csv(path / f"{i}_t{t}.csv", index_col=0) for t in [1, 2]
        ]
    )
    return labels


def reshape_dataset(loaded):
    """Reshape list of input data into two dataframes with each sample data in
    rows."""
    t1_data = loaded["data"].apply(
        lambda x: pd.Series(
            np.reshape(
                x[0].values if isinstance(x[0], pd.DataFrame) else x[0], -1
            )
        )
    )
    t2_data = loaded["data"].apply(
        lambda x: pd.Series(
            np.reshape(
                x[1].values if isinstance(x[0], pd.DataFrame) else x[0], -1
            )
        )
    )
    return t1_data, t2_data, loaded["group"]


def reshape_dataset_2d(dataset, m=10, n=10):
    """Reshape dataset into numpy matrix."""

    t1_data = dataset["data"].apply(
        lambda t: np.reshape(
            t[0].values if isinstance(t[0], pd.DataFrame) else t[0], (m, n, -1)
        )
    )
    t2_data = dataset["data"].apply(
        lambda t: np.reshape(
            t[1].values if isinstance(t[1], pd.DataFrame) else t[1], (m, n, -1)
        )
    )

    return np.stack(t1_data), np.stack(t2_data), dataset["group"]


def decomposition(dataset):
    """Non-linear decompositions of the data for visualization purposes."""
    t1, t2, y = reshape_dataset(dataset)
    # use a mds model first
    model = manifold.MDS()
    tf1 = model.fit_transform(t1, y)

    model = manifold.MDS()
    tf2 = model.fit_transform(t2, y)

    return tf1, tf2, y


def create_model_convolutional(xshape, yshape, num_inputs=2):
    """Create a convnet model. The data will be feeded as a 3d matrix."""
    inputs = []
    input_ends = []
    for i in range(num_inputs):
        t_input = layers.Input(shape=xshape)

        t_c1 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            strides=1,
        )(t_input)

        t_c2 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            strides=1,
        )(t_c1)

        t_p1 = layers.MaxPooling2D(
            pool_size=2, strides=1
        )(t_c2)

        t_d = layers.Dropout(0.25)(t_p1)

        t_f = layers.Flatten()(t_p1)

        t_end = t_f

        input_ends.append(t_end)
        inputs.append(t_input)

    concat = layers.concatenate(input_ends)

    m_a = layers.Dense(
        units=256, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(0.01)
    )(concat)
    # m_end = layers.Dense(
    #     units=64, activation="relu", kernel_initializer="uniform",
    #     kernel_regularizer=regularizers.l2(0.01)
    # )(m_a)
    m_end = layers.Dropout(0.5)(m_a)

    final = layers.Dense(
        units=yshape, activation="softmax"
    )(m_end)

    model = models.Model(inputs=inputs, outputs=final)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
    )

    return model


def create_model(xshape, yshape, num_inputs=2):
    """Create a simple sequential neural network with multiple inputs."""

    inputs = []
    input_ends = []
    for i in range(num_inputs):
        t_input = layers.Input(shape=xshape)

        t_attention = layers.Dense(
            units=xshape[0], activation="softmax",
            kernel_regularizer=regularizers.l2(.01)
        )(t_input)
        t_multatt = layers.multiply([t_input, t_attention])

        t_a = layers.Dense(
            units=128, activation="relu", kernel_initializer="uniform",
            # kernel_regularizer=regularizers.l1(.01)
        )(t_multatt)
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

    # t1_a = layers.Dense(
    #     units=128, activation="relu", kernel_initializer="uniform"
    # )(t1input)
    # t1_end = layers.Dense(
    #     units=64, activation="relu", kernel_initializer="uniform"
    # )(t1_a)

    # t2_a = layers.Dense(
    #     units=128, activation="relu", kernel_initializer="uniform"
    # )(t2input)
    # t2_end = layers.Dense(
    #     units=64, activation="relu", kernel_initializer="uniform"
    # )(t2_a)

    concat = layers.concatenate(input_ends)

    # m_a = layers.Dense(
    #     units=256, activation="relu", kernel_initializer="uniform"
    # )(concat)
    # m_b = layers.Dense(
    #     units=128, activation="relu", kernel_initializer="uniform"
    # )(m_a)
    m_end = layers.Dense(
        units=64, activation="relu", kernel_initializer="uniform"
    )(concat)

    final = layers.Dense(
        units=yshape, activation="softmax"
    )(m_end)

    model = models.Model(inputs=inputs, outputs=final)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
    )

    return model


def classify(data, m=10, n=10):
    """Extremely simple sequential neural network with two
    inputs for the 10x10x12 data

    m, n - denote the shape of the input
    """

    train, test = model_selection.train_test_split(data, train_size=0.8)

    tr1, tr2, ytrain = reshape_dataset(train)

    binarizer = preprocessing.LabelBinarizer()
    ytrain_mat = binarizer.fit_transform(ytrain)

    te1, te2, ytest = reshape_dataset(test)
    ytest_mat = binarizer.transform(ytest)

    # model = naive_bayes.GaussianNB()
    # model.fit(tr1, ytrain)

    model = create_model((tr1.shape[1], ), ytrain_mat.shape[1])
    model.fit([tr1, tr2], ytrain_mat, epochs=20, batch_size=16)
    pred = model.predict([te1, te2], batch_size=128)
    pred = binarizer.inverse_transform(pred)

    # res = model.score(te1, ytest)
    # print(res)
    # pred = model.predict(te1)

    print("F1: ", metrics.f1_score(ytest, pred, average="micro"))

    confusion = metrics.confusion_matrix(ytest, pred, binarizer.classes_,)
    return confusion, binarizer.classes_


def classify_convolutional(data, m=10, n=10):
    train, test = model_selection.train_test_split(data, train_size=0.8)

    tr1, tr2, ytrain = reshape_dataset_2d(train, m=m, n=n)

    binarizer = preprocessing.LabelBinarizer()
    ytrain_mat = binarizer.fit_transform(ytrain)

    te1, te2, ytest = reshape_dataset_2d(test, m=m, n=n)
    ytest_mat = binarizer.transform(ytest)

    model = create_model_convolutional(tr1[0].shape, ytrain_mat.shape[1])
    model.fit([tr1, tr2], ytrain_mat, epochs=30, batch_size=16)
    pred = model.predict([te1, te2], batch_size=128)
    pred = binarizer.inverse_transform(pred)

    print("F1: ", metrics.f1_score(ytest, pred, average="micro"))

    confusion = metrics.confusion_matrix(ytest, pred, binarizer.classes_,)
    return confusion, binarizer.classes_


def subtract_ref_data(data, references):
    data["data"] = data["data"].apply(
        lambda t: [r - a for r, a in zip(references, t)]
    )
    return data


def normalize_data(data):
    data["data"] = data["data"].apply(
        lambda t: [preprocessing.StandardScaler().fit_transform(d) for d in t]
    )
    return data


def modify_groups(data, mapping):
    """Change the cohort composition according to the given
    cohort composition."""
    data["group"] = data["group"].apply(lambda g: mapping.get(g, g))
    return data

def main():

    inputpath = pathlib.Path("sommaps/huge_s30")

    indata = load_dataset(inputpath)

    # ref_maps = [
    #     pd.read_csv(f"sommaps/reference/t{t}.csv", index_col=0)
    #     for t in [1, 2]
    # ]
    # subdata = subtract_ref_data(indata, ref_maps)

    group_map = {
        "CLL": "CM",
        "MBL": "CM",
        "MZL": "LM",
        "LPL": "LM",
        "MCL": "MP",
        "PL": "MP",
    }

    mapped_data = modify_groups(indata, mapping=group_map)

    normdata = normalize_data(indata)
    mapped_normdata = normalize_data(mapped_data)

    # plotpath = pathlib.Path("sommaps/output/lotta")
    # tf1, tf2, y = decomposition(indata)
    # plot_transformed(plotpath, tf1, tf2, y)

    confusion, groups = classify(normdata)
    confusion, groups = classify_convolutional(normdata, m=30, n=30)

    outpath = pathlib.Path("sommaps/output/huge_s30")
    outpath.mkdir(parents=True, exist_ok=True)

    plotting.plot_confusion_matrix(
        confusion, groups, normalize=True,
        filename=outpath / "confusion_merged", dendroname=outpath / "dendro_merged"
    )


if __name__ == "__main__":
    main()
