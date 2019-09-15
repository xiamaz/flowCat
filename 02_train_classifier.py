#!/usr/bin/env python3
"""
Train models on Munich data and attempt to classify Bonn data.
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models  # pylint: disable=import-error
# from keras import layers, regularizers, models
from argmagic import argmagic

from flowcat import utils, io_functions, mappings, classification_utils
from flowcat.som_dataset import SOMDataset, SOMSequence


def create_model_early_merge(input_shapes, yshape, global_decay=5e-6):
    inputs = []
    for xshape in input_shapes:
        ix = layers.Input(shape=xshape)
        inputs.append(ix)

    x = layers.concatenate(inputs)
    x = layers.Conv2D(
        filters=64, kernel_size=4, activation="relu", strides=3,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Conv2D(
        filters=96, kernel_size=3, activation="relu", strides=2,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Conv2D(
        filters=128, kernel_size=1, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # x = layers.Dropout(0.2)(x)

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

    model = create_model_multi_input(inputs, output, **kwargs)
    # model = create_model_early_merge(inputs, output, **kwargs)

    binarizer = LabelBinarizer()
    binarizer.fit(groups)
    return binarizer, model


def main(data: utils.URLPath, meta: utils.URLPath, output: utils.URLPath):
    """
    Args:
        data: Path to som dataset
        output: Output path
    """
    tubes = ("1", "2")
    pad_width = 1

    group_mapping = mappings.GROUP_MAPS["8class"]
    # mapping = group_mapping["map"]
    mapping = None
    # groups = group_mapping["groups"]
    groups = mappings.GROUPS

    # dataset = io_functions.load_case_collection(data, meta)
    dataset = SOMDataset.from_path(data)
    if mapping:
        dataset = dataset.map_groups(mapping)

    dataset = dataset.filter(groups=groups)

    dataset_groups = {d.group for d in dataset}

    if set(groups) != dataset_groups:
        raise RuntimeError(f"Group mismatch: {groups}, but got {dataset_groups}")

    validate, train = dataset.create_split(50, stratify=True)

    group_count = train.group_count
    group_weights = classification_utils.calculate_group_weights(group_count)
    group_weights = {
        i: group_weights.get(g, 1.0) for i, g in enumerate(groups)
    }

    # train = train.balance(2000)
    # train = train.balance_per_group({
    #     "CM": 6000,
    #     # "CLL": 4000,
    #     # "MBL": 2000,
    #     "MCL": 1000,
    #     "PL": 1000,
    #     "LPL": 1000,
    #     "MZL": 1000,
    #     "FL": 1000,
    #     "HCL": 1000,
    #     "normal": 6000,
    # })

    io_functions.save_json(train.labels, output / "ids_train.json")
    io_functions.save_json(validate.labels, output / "ids_validate.json")

    som_config = io_functions.load_json(data + "_config.json")
    selected_tubes = {tube: som_config[tube] for tube in tubes}

    config = {
        "tubes": selected_tubes,
        "groups": groups,
        "pad_width": pad_width,
        "mapping": group_mapping,
    }
    io_functions.save_json(config, output / "config.json")

    for tube in tubes:
        x, y, z = selected_tubes[tube]["dims"]
        selected_tubes[tube]["dims"] = (x + 2 * pad_width, y + 2 * pad_width, z)

    cost_mapping = {
        ("CLL", "MBL"): 0.5,
        ("MBL", "CLL"): 0.5,
        ("MCL", "PL"): 0.5,
        ("PL", "MCL"): 0.5,
        ("LPL", "MZL"): 0.5,
        ("MZL", "LPL"): 0.5,
        ("CLL", "normal"): 2,
        ("MBL", "normal"): 2,
        ("MCL", "normal"): 2,
        ("PL", "normal"): 2,
        ("LPL", "normal"): 2,
        ("MZL", "normal"): 2,
        ("FL", "normal"): 2,
        ("HCL", "normal"): 2,
    }
    cost_matrix = classification_utils.build_cost_matrix(cost_mapping, groups)

    binarizer, model = get_model(selected_tubes, groups=groups, global_decay=5e-7)

    model.compile(
        loss=classification_utils.WeightedCategoricalCrossentropy(cost_matrix),
        # loss="categorical_crossentropy",
        # loss="binary_crossentropy",
        optimizer="adam",
        # optimizer=optimizers.Adam(lr=0.0, decay=0.0, epsilon=epsilon),
        metrics=[
            "acc",
        ]
    )

    def getter_fun(sample, tube):
        return sample.get_tube(tube)

    trainseq = SOMSequence(
        train,
        binarizer,
        tube=tubes,
        get_array_fun=getter_fun,
        batch_size=32,
        pad_width=pad_width)
    validseq = SOMSequence(
        validate,
        binarizer,
        tube=tubes,
        get_array_fun=getter_fun,
        batch_size=128,
        pad_width=pad_width)

    tensorboard_dir = str(output / "tensorboard")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=5,
        write_grads=True,
        write_images=True,
    )
    nan_callback = keras.callbacks.TerminateOnNaN()

    model.fit_generator(
        epochs=15, shuffle=True,
        callbacks=[tensorboard_callback, nan_callback],
        class_weight=group_weights,
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
    confusion = pd.DataFrame(confusion, index=groups, columns=groups)
    print(confusion)
    io_functions.save_csv(confusion, output / "validation_confusion.csv")
    metrics_results = {
        "balanced": metrics.balanced_accuracy_score(true_labels, pred_labels),
        "f1_micro": metrics.f1_score(true_labels, pred_labels, average="micro"),
        "f1_macro": metrics.f1_score(true_labels, pred_labels, average="macro"),
        "mcc": metrics.matthews_corrcoef(true_labels, pred_labels),
    }
    print(metrics_results)
    io_functions.save_json(metrics_results, output / "validation_metrics.json")

    # preds = []
    # for pred in model.predict_generator(validseq):
    #     preds.append(pred)

    # args.output.local.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    argmagic(main)
