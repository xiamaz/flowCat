#!/usr/bin/env python3
"""
Train models on Munich data and attempt to classify Bonn data.
"""
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from keras import layers, regularizers, models
from argmagic import argmagic

from flowcat import utils, io_functions, mappings
from flowcat.som_dataset import SOMDataset, SOMSequence


def create_model_multi_input(input_shapes, yshape, global_decay=5e-4):
    segments = []
    inputs = []
    print(input_shapes)
    for xshape in input_shapes:
        ix = layers.Input(shape=xshape)
        inputs.append(ix)
        x = layers.Conv2D(
            filters=32, kernel_size=2, activation="relu", strides=1,
            # kernel_regularizer=regularizers.l2(global_decay),
        )(ix)
        x = layers.Conv2D(
            filters=32, kernel_size=2, activation="relu", strides=1,
            # kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
        x = layers.Conv2D(
            filters=32, kernel_size=2, activation="relu", strides=1,
            # kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        x = layers.Conv2D(
            filters=32, kernel_size=2, activation="relu", strides=1,
            # kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
        x = layers.GlobalAveragePooling2D()(x)
        segments.append(x)

    x = layers.average(segments)
    # x = layers.Conv2D(
    #     filters=32, kernel_size=2, activation="relu", strides=1,
    #     kernel_regularizer=regularizers.l2(global_decay))(x)
    # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    # x = layers.Flatten()(ix)

    # x = layers.Dense(
    #     units=128, activation="relu", kernel_initializer="uniform",
    #     kernel_regularizer=regularizers.l2(global_decay)
    # )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=128, activation="relu", kernel_initializer="uniform",
        # kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    x = layers.Dense(
        units=64, activation="relu", kernel_initializer="uniform",
        # kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    final = layers.Dense(
        units=yshape, activation="sigmoid"
    )(x)

    model = models.Model(inputs=inputs, outputs=final)
    for layer in model.layers:
        print(layer.output_shape)
    return model


def get_model(channel_config, groups, **kwargs):
    inputs = tuple([*d["dims"][:-1], len(d["channels"])] for d in channel_config.values())
    output = len(groups)

    model = create_model_multi_input(inputs, output, **kwargs)
    model.compile(
        loss="categorical_crossentropy",
        # loss="binary_crossentropy",
        optimizer="adam",
        # optimizer=optimizers.Adam(lr=0.0, decay=0.0, epsilon=epsilon),
        metrics=[
            "acc",
            # top2_acc,
        ]
    )

    binarizer = LabelBinarizer()
    binarizer.fit(groups)
    return binarizer, model


def main(data: utils.URLPath, output: utils.URLPath):
    """
    Args:
        data: Path to som dataset
        output: Output path
    """
    tubes = ("1", "2")

    munich = SOMDataset.from_path(data)
    train, validate = munich.split(ratio=0.9, stratified=True)

    train_ids = [t.label for t in train.data]
    validate_ids = [t.label for t in validate.data]
    io_functions.save_json(train_ids, output / "ids_train.json")
    io_functions.save_json(validate_ids, output / "ids_validate.json")

    selected_tubes = {tube: munich.config[tube] for tube in tubes}
    groups = mappings.GROUPS

    config = {
        "tubes": selected_tubes,
        "groups": groups,
    }
    io_functions.save_json(config, output / "config.json")

    binarizer, model = get_model(selected_tubes, mappings.GROUPS, global_decay=5e-6)

    trainseq = SOMSequence(train, binarizer, tube=tubes)
    validseq = SOMSequence(validate, binarizer, tube=tubes)

    tensorboard_dir = str(output / "tensorboard")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=5,
        write_grads=True,
        write_images=True,
    )

    model.fit_generator(
        epochs=100, shuffle=True,
        callbacks=[tensorboard_callback],
        generator=trainseq, validation_data=validseq)

    model.save(str(output / "model.h5"))
    io_functions.save_joblib(binarizer, output / "binarizer.joblib")

    # preds = []
    # for pred in model.predict_generator(validseq):
    #     preds.append(pred)

    # args.output.local.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    argmagic(main)
