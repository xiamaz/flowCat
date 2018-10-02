"""
Model using FCS input.
"""

from keras import layers, regularizers, models


def fcs_1d_avgpool(x, global_decay=5e-4):
    """1x1 convolutions on raw FCS data."""
    x = layers.Conv1D(
        50, 1, strides=1, activation="elu",
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Conv1D(
        50, 1, strides=1, activation="elu",
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Conv1D(
        50, 1, strides=1, activation="elu",
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.BatchNormalization()(x)

    return x


def fcs_1d_maxpool(x, global_decay=5e-4):
    """1x1 convolutions, but using maxpool instead of avgpool.
    We probably still need something smarter.
    """
    x = layers.Conv1D(16, 1, strides=1, activation="elu")(x)
    x = layers.Conv1D(8, 1, strides=1, activation="elu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.BatchNormalization()(x)
    return x


def create_model_fcs(xshape, yshape, global_decay=5e-4):
    """Create direct FCS classification model."""
    xinput = layers.Input(shape=xshape[0])

    x = fcs_1d_avgpool(xinput)
    # x = layers.concatenate([xa, xb])

    x = layers.Dense(
        100, activation="elu",
        kernel_regularizer=regularizers.l2(global_decay))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        100, activation="elu",
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Dense(
        50, activation="elu",
        kernel_regularizer=regularizers.l2(global_decay))(x)
    # x = layers.Dense(32, activation="elu")(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(16)(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(16)(x)
    # x = layers.Dropout(0.2)(x)

    final = layers.Dense(yshape, activation="softmax")(x)
    model = models.Model(inputs=[xinput], outputs=final)
    return model
