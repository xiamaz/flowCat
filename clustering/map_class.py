import pathlib

import numpy as np
import pandas as pd
from sklearn import manifold, model_selection, preprocessing
from sklearn import naive_bayes
from sklearn import metrics

from keras import layers, models

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


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


def decomposition(dataset):
    """Non-linear decompositions of the data for visualization purposes."""
    t1, t2, y = reshape_dataset(dataset)
    # use a mds model first
    model = manifold.MDS()
    tf1 = model.fit_transform(t1, y)

    model = manifold.MDS()
    tf2 = model.fit_transform(t2, y)

    return tf1, tf2, y


def create_model(xshape, yshape):
    """Create a simple sequential neural network with multiple inputs."""

    t1input = layers.Input(shape=xshape)
    t1_a = layers.Dense(
        units=512, activation="relu", kernel_initializer="uniform"
    )(t1input)
    t1_end = layers.Dense(
        units=256, activation="relu", kernel_initializer="uniform"
    )(t1_a)

    t2input = layers.Input(shape=xshape)
    t2_a = layers.Dense(
        units=512, activation="relu", kernel_initializer="uniform"
    )(t2input)
    t2_end = layers.Dense(
        units=256, activation="relu", kernel_initializer="uniform"
    )(t2_a)

    concat = layers.concatenate([t1_end, t2_end])

    m_a = layers.Dense(
        units=1024, activation="relu", kernel_initializer="uniform"
    )(concat)
    m_b = layers.Dense(
        units=256, activation="relu", kernel_initializer="uniform"
    )(m_a)
    m_end = layers.Dense(
        units=64, activation="relu", kernel_initializer="uniform"
    )(m_b)

    final = layers.Dense(
        units=yshape, activation="softmax"
    )(m_end)

    model = models.Model(inputs=[t1input, t2input], outputs=final)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
    )

    return model


def classify(data):
    """Extremely simple sequential neural network with two
    inputs for the 10x10x12 data"""

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


def main():

    ref_maps = [
        pd.read_csv(f"sommaps/reference/t{t}.csv", index_col=0) for t in [1, 2]
    ]
    inputpath = pathlib.Path("sommaps/lotta")

    indata = load_dataset(inputpath)

    subdata = subtract_ref_data(indata, ref_maps)

    normdata = normalize_data(indata)

    # plotpath = pathlib.Path("sommaps/output/lotta")
    # tf1, tf2, y = decomposition(indata)
    # plot_transformed(plotpath, tf1, tf2, y)

    classify(normdata)


if __name__ == "__main__":
    main()
