"""
Train models on Munich data and attempt to classify Bonn data.
"""
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import TSNE
from keras import layers, regularizers, models
from keras.utils import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowcat.som_dataset import SOMDataset


def create_model(xshape, yshape, global_decay=5e-4):
    ix = layers.Input(shape=xshape)

    # x = layers.Conv2D(
    #     filters=32, kernel_size=2, activation="relu", strides=1,
    #     kernel_regularizer=regularizers.l2(global_decay))(ix)
    # x = layers.Conv2D(
    #     filters=32, kernel_size=2, activation="relu", strides=1,
    #     kernel_regularizer=regularizers.l2(global_decay))(x)
    # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

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
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=ix, outputs=final)
    return model


class SOMSequence(Sequence):

    def __init__(self, data: SOMDataset, binarizer, batch_size: int = 32, tube=1):
        self.data = data
        self.tube = tube
        self.batch_size = batch_size
        self.binarizer = binarizer

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = np.array([s.get_tube(self.tube).np_array() for s in batch])
        y_batch = self.binarizer.transform([s.group for s in batch])
        return x_batch, y_batch


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


def main():
    munich = SOMDataset.from_path("output/01b-create-soms/testmll")
    bonn = SOMDataset.from_path("output/01b-create-soms/testbonn")

    # visualize tsne first
    visualize_tsne(munich, "testmunich.png")
    visualize_tsne(bonn, "testbonn.png")
    return

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

    trainseq = SOMSequence(train, binarizer)
    validseq = SOMSequence(validate, binarizer)

    model.fit_generator(
        epochs=10,
        generator=trainseq, validation_data=validseq)


if __name__ == "__main__":
    main()
