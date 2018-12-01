"""
Classifiers for classification to label level
"""

from keras import layers, regularizers, models

from . import weighted_crossentropy


def sommap_tube(x, global_decay=5e-4):
    """Block to process a single tube."""
    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="relu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    return x


def sommap_merged(t1, t2, global_decay=5e-4):
    """Processing of SOM maps using multiple tubes."""
    t1 = sommap_tube(t1)
    t2 = sommap_tube(t2)
    # x = layers.concatenate([t1, t2])
    x = layers.average([t1, t2])

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
    return x


def create_model_cnn(xshape, yshape):
    """Create a convnet model. The data will be feeded as a 3d matrix."""
    t1 = layers.Input(shape=xshape[0])
    t2 = layers.Input(shape=xshape[1])
    x = sommap_merged(t1, t2)

    final = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=[t1, t2], outputs=final)

    return model


# ALTERNATIVE STUFF
def sommap_tube_elu_globalavg(x, global_decay=5e-4):
    """Block to process a single tube."""
    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="elu", strides=2,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="elu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.MaxPooling2D(pool_size=2, strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="elu", strides=1,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.Conv2D(
        filters=32, kernel_size=2, activation="elu", strides=2,
        kernel_regularizer=regularizers.l2(global_decay))(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # x = layers.Flatten()(x)
    return x


def sommap_merged_elu_globalavg(t1, t2, global_decay=5e-4):
    """Processing of SOM maps using multiple tubes."""
    t1 = sommap_tube_elu_globalavg(t1, global_decay=global_decay)
    t2 = sommap_tube_elu_globalavg(t2, global_decay=global_decay)
    # x = layers.concatenate([t1, t2])
    x = layers.multiply([t1, t2])

    x = layers.Dense(
        units=128, activation="elu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        units=64, activation="elu", kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    return x
