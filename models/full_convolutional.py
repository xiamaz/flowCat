# pylint: skip-file
# flake8: noqa
import keras
import keras.backend as K
from keras import layers, regularizers, models


def create_model(input_shapes, yshape, global_decay=5e-6):
    segments = []
    inputs = []
    for xshape in input_shapes:
        ix = layers.Input(shape=xshape)
        inputs.append(ix)
        x = layers.Conv2D(
            filters=32, kernel_size=4, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(ix)
        # x = layers.Conv2D(
        #     filters=32, kernel_size=3, activation="relu", strides=1,
        #     kernel_regularizer=regularizers.l2(global_decay),
        # )(x)
        x = layers.Conv2D(
            filters=48, kernel_size=2, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        x = layers.Conv2D(
            filters=64, kernel_size=2, activation="relu", strides=2,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

        # # x = layers.GlobalAveragePooling2D()(x)
        x = layers.GlobalMaxPooling2D()(x)
        segments.append(x)

    # x = layers.concatenate(segments, axis=-1)
    x = layers.Lambda(lambda x: K.stack(x, axis=-1))(segments)
    x = layers.Conv1D(
        filters=64, kernel_size=64, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay),
    )(x)
    x = layers.Conv1D(
        filters=32, kernel_size=1, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay),
    )(x)
    x = layers.Flatten()(x)
    # x = layers.Conv1D(
    #     filters=64, kernel_size=2, activation="relu", strides=1,
    #     kernel_regularizer=regularizers.l2(global_decay),
    # )(x)
    # x = layers.GlobalMaxPooling1D()(x)

    # x = layers.Dense(
    #     units=64, activation="relu",
    #     # kernel_initializer="uniform",
    #     kernel_regularizer=regularizers.l2(global_decay)
    # )(x)
    # x = layers.Dense(
    #     units=32, activation="relu",
    #     # kernel_initializer="uniform",
    #     kernel_regularizer=regularizers.l2(global_decay)
    # )(x)

    x = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=inputs, outputs=x)
    model.summary()
    return model
