#!/usr/bin/env python3
"""
Train models on Munich data and attempt to classify Bonn data.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models  # pylint: disable=import-error
from argmagic import argmagic

from flowcat import utils, io_functions, mappings, fcs_dataset, classification_utils
from flowcat.dataset import case_dataset


class FCSSequence(keras.utils.Sequence):
    def __init__(self, dataset, binarizer, tubes, sample_size=128, batch_size=32):
        """Create batches of data for multiinput model."""
        self.dataset = dataset
        self.binarizer = binarizer
        self.batch_size = batch_size
        self.tubes = tubes
        self.sample_size = sample_size
        self._cache = {}

    @property
    def true_labels(self):
        return [d.group for d in self.dataset]

    @property
    def xshape(self):
        return [
            (self.sample_size, len(self.dataset.selected_markers[tube]))
            for tube in self.tubes
        ]

    @property
    def yshape(self):
        return len(self.binarizer.classes_)

    def _get_fcs_data(self, case, tube):
        fcsdata = case.get_tube(tube).get_data()
        markers = self.dataset.selected_markers[tube]
        arr = fcsdata.data[markers].values
        arr = arr[np.random.randint(arr.shape[0], size=self.sample_size), :]
        return arr

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        inputs = []
        batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
        for tube in self.tubes:
            data = np.array([self._get_fcs_data(c, tube) for c in batch])
            inputs.append(data)

        y_labels = [s.group for s in batch]
        y_batch = self.binarizer.transform(y_labels)

        self._cache[idx] = inputs, y_batch

        return inputs, y_batch


def create_fcs_model(xshape, yshape, global_decay=5e-6):
    segments = []
    inputs = []
    for shape in xshape:
        ix = layers.Input(shape=shape)
        inputs.append(ix)
        x = layers.BatchNormalization()(ix)
        x = layers.Conv1D(
            filters=64, kernel_size=1, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        x = layers.Conv1D(
            filters=64, kernel_size=1, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        x = layers.Conv1D(
            filters=64, kernel_size=1, activation="relu", strides=1,
        )(x)
        x = layers.Conv1D(
            filters=64, kernel_size=1, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        # x = layers.GlobalAveragePooling1D()(x)
        x = layers.GlobalMaxPooling1D()(x)
        segments.append(x)
    x = layers.concatenate(segments)
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
    x = layers.Dense(
        units=yshape, activation="softmax"
    )(x)
    model = models.Model(inputs=inputs, outputs=x)
    return model


def main(data: utils.URLPath, meta: utils.URLPath, output: utils.URLPath):
    """
    Args:
        data: Path to fcs dataset data
        meta: Path to fcs dataset metainformation
        output: Output path
    """
    tubes = ("1", "2")
    sample_size = 4096
    # group_mapping = mappings.GROUP_MAPS["6class"]
    # mapping = group_mapping["map"]
    mapping = None
    groups = mappings.GROUPS
    # groups = group_mapping["groups"]

    dataset = io_functions.load_case_collection(data, meta)
    if mapping:
        dataset = dataset.map_groups(mapping)
    dataset = dataset.filter(groups=groups)

    validate, train = dataset.create_split(50)
    print(train.group_count)
    # train = train.balance(1000).shuffle()
    train = train.sample(2000).shuffle()
    print(train.group_count)

    group_count = train.group_count
    group_weights = classification_utils.calculate_group_weights(group_count)
    group_weights = {
        i: group_weights.get(g, 1.0) for i, g in enumerate(groups)
    }

    io_functions.save_json(train.labels, output / "ids_train.json")
    io_functions.save_json(validate.labels, output / "ids_validate.json")

    binarizer = LabelBinarizer()
    binarizer.fit(groups)

    train_seq = FCSSequence(
        train, binarizer, tubes=tubes, sample_size=sample_size, batch_size=64
    )
    validate_seq = FCSSequence(
        validate, binarizer, tubes=tubes, sample_size=sample_size, batch_size=128
    )

    config = {
        "tubes": tubes,
        "groups": groups,
    }
    io_functions.save_json(config, output / "config.json")

    # for tube in tubes:
    #     x, y, z = selected_tubes[tube]["dims"]
    #     selected_tubes[tube]["dims"] = (x + 2 * pad_width, y + 2 * pad_width, z)

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

    model = create_fcs_model(train_seq.xshape, train_seq.yshape, global_decay=5e-5)
    model.compile(
        # loss="categorical_crossentropy",
        # loss=keras.losses.CategoricalCrossentropy(),
        loss=classification_utils.WeightedCategoricalCrossentropy(cost_matrix),
        # loss="binary_crossentropy",
        optimizer="adam",
        # optimizer=optimizers.Adam(lr=0.0, decay=0.0, epsilon=epsilon),
        metrics=[
            "acc",
            # keras.metrics.CategoricalAccuracy(),
            # keras.metrics.TopKCategoricalAccuracy(k=2),
            # top2_acc,
        ]
    )
    model.summary()

    tensorboard_dir = str(output / "tensorboard")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=5,
        write_grads=True,
        write_images=True,
    )
    nan_callback = keras.callbacks.TerminateOnNaN()

    model.fit_generator(
        epochs=20, shuffle=True,
        callbacks=[
            tensorboard_callback,
            nan_callback
        ],
        class_weight=group_weights,
        generator=train_seq, validation_data=validate_seq)

    model.save(str(output / "model.h5"))
    io_functions.save_joblib(binarizer, output / "binarizer.joblib")

    preds = []
    for pred in model.predict_generator(validate_seq):
        preds.append(pred)
    pred_arr = np.array(preds)
    pred_labels = binarizer.inverse_transform(pred_arr)
    true_labels = validate_seq.true_labels

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
