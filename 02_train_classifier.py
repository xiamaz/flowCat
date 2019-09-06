#!/usr/bin/env python3
"""
Train models on Munich data and attempt to classify Bonn data.
"""
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models  # pylint: disable=import-error
# from keras import layers, regularizers, models
from argmagic import argmagic

from flowcat import utils, io_functions, mappings
from flowcat.som_dataset import SOMDataset, SOMSequence


def create_model_early_merge(input_shapes, yshape, global_decay=5e-6):
    inputs = []
    for xshape in input_shapes:
        ix = layers.Input(shape=xshape)
        inputs.append(ix)

    x = layers.concatenate(inputs)
    # x = layers.Conv2D(
    #     filters=32, kernel_size=2, activation="relu", strides=1,
    #     kernel_regularizer=regularizers.l2(global_decay))(x)
    # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(ix)

    x = layers.Dense(
        units=128, activation="relu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=128, activation="relu",
        # kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(
        units=64, activation="relu",
        # kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=inputs, outputs=x)
    for layer in model.layers:
        print(layer.output_shape)
    return model


def create_model_multi_input(input_shapes, yshape, global_decay=5e-6):
    segments = []
    inputs = []
    print(input_shapes)
    for xshape in input_shapes:
        ix = layers.Input(shape=xshape)
        inputs.append(ix)
        x = layers.Conv2D(
            filters=32, kernel_size=4, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(ix)
        x = layers.Conv2D(
            filters=48, kernel_size=3, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        # x = layers.Conv2D(
        #     filters=32, kernel_size=2, activation="relu", strides=1,
        #     kernel_regularizer=regularizers.l2(global_decay),
        # )(x)
        # x = layers.Conv2D(
        #     filters=64, kernel_size=2, activation="relu", strides=1,
        #     # kernel_regularizer=regularizers.l2(global_decay),
        # )(x)
        # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
        x = layers.Conv2D(
            filters=48, kernel_size=2, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        x = layers.Conv2D(
            filters=64, kernel_size=2, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

        # x = layers.GlobalAveragePooling2D()(x)
        x = layers.GlobalMaxPooling2D()(x)
        segments.append(x)

    x = layers.concatenate(segments)
    # x = layers.Conv2D(
    #     filters=32, kernel_size=2, activation="relu", strides=1,
    #     kernel_regularizer=regularizers.l2(global_decay))(x)
    # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # x = layers.Dropout(0.2)(x)

    # x = layers.Flatten()(ix)

    # x = layers.Dense(
    #     units=128, activation="relu", kernel_initializer="uniform",
    #     kernel_regularizer=regularizers.l2(global_decay)
    # )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=128, activation="relu",
        # kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(
        units=64, activation="relu",
        # kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=inputs, outputs=x)
    for layer in model.layers:
        print(layer.output_shape)
    return model


def get_model(channel_config, groups, **kwargs):
    inputs = tuple([*d["dims"][:-1], len(d["channels"])] for d in channel_config.values())
    output = len(groups)

    # model = create_model_multi_input(inputs, output, **kwargs)
    model = create_model_early_merge(inputs, output, **kwargs)
    model.compile(
        loss="categorical_crossentropy",
        # loss="binary_crossentropy",
        optimizer="adam",
        # optimizer=optimizers.Adam(lr=0.0, decay=0.0, epsilon=epsilon),
        metrics=[
            "acc",
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
    pad_width = 1

    group_mapping = mappings.GROUP_MAPS["8class"]
    mapping = group_mapping["map"]
    groups = group_mapping["groups"]

    dataset = SOMDataset.from_path(data)
    if mapping:
        dataset = dataset.map_groups(mapping)

    dataset = dataset.filter_groups(groups)

    dataset_groups = {d.group for d in dataset.data}

    if set(groups) != dataset_groups:
        raise RuntimeError(f"Group mismatch: {groups}, but got {dataset_groups}")

    train, validate = dataset.split(ratio=0.9, stratified=True)

    # train = train.balance(2000)
    train = train.balance_per_group({
        "CM": 6000,
        # "CLL": 4000,
        # "MBL": 2000,
        "MCL": 1000,
        "PL": 1000,
        "LPL": 1000,
        "MZL": 1000,
        "FL": 1000,
        "HCL": 1000,
        "normal": 6000,
    })

    train_ids = [t.label for t in train.data]
    validate_ids = [t.label for t in validate.data]
    io_functions.save_json(train_ids, output / "ids_train.json")
    io_functions.save_json(validate_ids, output / "ids_validate.json")

    selected_tubes = {tube: dataset.config[tube] for tube in tubes}

    config = {
        "tubes": selected_tubes,
        "groups": groups,
    }
    io_functions.save_json(config, output / "config.json")

    for tube in tubes:
        x, y, z = selected_tubes[tube]["dims"]
        selected_tubes[tube]["dims"] = (x + 2 * pad_width, y + 2 * pad_width, z)

    binarizer, model = get_model(selected_tubes, groups=groups, global_decay=5e-7)

    trainseq = SOMSequence(train, binarizer, tube=tubes, batch_size=32, pad_width=pad_width)
    validseq = SOMSequence(validate, binarizer, tube=tubes, batch_size=128, pad_width=pad_width)

    tensorboard_dir = str(output / "tensorboard")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=5,
        write_grads=True,
        write_images=True,
    )
    nan_callback = keras.callbacks.TerminateOnNaN()

    model.fit_generator(
        epochs=100, shuffle=True,
        callbacks=[tensorboard_callback, nan_callback],
        generator=trainseq, validation_data=validseq)

    model.save(str(output / "model.h5"))
    io_functions.save_joblib(binarizer, output / "binarizer.joblib")

    preds = []
    for pred in model.predict_generator(validseq):
        preds.append(pred)
    pred_arr = np.array(preds)
    pred_labels = binarizer.inverse_transform(pred_arr)
    true_labels = validseq.true_labels

    confusion = metrics.confusion_matrix(true_labels, pred_labels, labels=groups)
    print(groups)
    print(confusion)
    balanced = metrics.balanced_accuracy_score(true_labels, pred_labels)
    print(balanced)

    # preds = []
    # for pred in model.predict_generator(validseq):
    #     preds.append(pred)

    # args.output.local.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    argmagic(main)
