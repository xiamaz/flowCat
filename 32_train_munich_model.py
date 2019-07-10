"""
Train models on Munich data and attempt to classify Bonn data.
"""
from argparse import ArgumentParser

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import TSNE
from keras import layers, regularizers, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowcat.som_dataset import SOMDataset
from flowcat import utils, SOMSequence


def create_model(xshape, yshape, global_decay=5e-4):
    ix = layers.Input(shape=xshape)

    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(ix)
    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(ix)

    x = layers.Dense(
        units=128, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=64, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    final = layers.Dense(
        units=yshape, activation="sigmoid"
    )(x)

    model = models.Model(inputs=ix, outputs=final)
    return model


def visualize_tsne(data, output):
    xdata = np.array([np.ravel(s.get_tube(1).data.values) for s in data.data])
    ydata = np.array([s.group for s in data.data])
    model = TSNE()
    result = model.fit_transform(xdata)
    fix, ax = plt.subplots()
    for group, color in [("CLL", "red"), ("normal", "blue")]:
        ax.scatter(
            result[ydata == group, 0],
            result[ydata == group, 1],
            label=group, color=color
        )
    ax.legend()
    ax.set_title("CLL and normal")
    fix.savefig(output)


def main(args):
    munich = SOMDataset.from_path(args.input)
    # bonn = SOMDataset.from_path("output/01b-create-soms/testbonn")

    # visualize tsne first
    # visualize_tsne(munich, "testmunich.png")
    # visualize_tsne(bonn, "testbonn.png")

    train, validate = munich.split(ratio=0.9, stratified=True)

    model = create_model(munich.dims, 1)

    model.compile(
        # loss="categorical_crossentropy",
        loss="binary_crossentropy",
        optimizer="adam",
        # optimizer=optimizers.Adam(lr=0.0, decay=0.0, epsilon=epsilon),
        metrics=[
            "acc",
            # top2_acc,
        ]
    )

    binarizer = LabelBinarizer()
    binarizer.fit(["CLL", "normal"])

    trainseq = SOMSequence(train, binarizer, tube=1)
    validseq = SOMSequence(validate, binarizer, tube=1)

    model.fit_generator(
        epochs=20,
        generator=trainseq, validation_data=validseq)

    args.output.local.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output / "model.h5"))


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("input", type=utils.URLPath)
    PARSER.add_argument("output", type=utils.URLPath)
    main(PARSER.parse_args())
