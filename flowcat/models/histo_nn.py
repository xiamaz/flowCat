from keras import layers, models, regularizers
import keras


def histogram_tube(x, global_decay):
    """Processing of histogram information using dense neural net."""
    x = layers.Dense(
        units=16, activation="elu", kernel_initializer="uniform",
        kernel_regularizer=keras.regularizers.l2(l=global_decay))(x)
    # x = layers.Dropout(rate=0.01)(x)
    x = layers.Dense(
        units=16, activation="elu",
        kernel_regularizer=regularizers.l2(l=global_decay))(x)
    # x = layers.Dropout(rate=0.01)(x)
    # x = layers.BatchNormalization()(x)
    return x


def histogram_merged(t1, t2, global_decay):
    """Overall merged processing of histogram information."""
    t1 = histogram_tube(t1, global_decay)
    t2 = histogram_tube(t2, global_decay)
    x = layers.concatenate([t1, t2])
    x = layers.Dense(
        units=16, activation="elu",
        kernel_regularizer=regularizers.l2(global_decay))(x)
    return x


def create_model_histo(xshape, yshape, global_decay=1e-5):
    """Create a simple sequential neural network with multiple inputs."""

    t1_input = layers.Input(shape=xshape[0])
    t2_input = layers.Input(shape=xshape[1])

    x = histogram_merged(t1_input, t2_input, global_decay)
    final = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=[t1_input, t2_input], outputs=final)

    return model
