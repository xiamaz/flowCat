


def histogram_tube(x):
    """Processing of histogram information using dense neural net."""
    x = layers.Dense(
        units=16, activation="elu", kernel_initializer="uniform",
        kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    # x = layers.Dropout(rate=0.01)(x)
    x = layers.Dense(
        units=16, activation="elu",
        kernel_regularizer=regularizers.l2(l=GLOBAL_DECAY))(x)
    # x = layers.Dropout(rate=0.01)(x)
    # x = layers.BatchNormalization()(x)
    return x


def histogram_merged(t1, t2):
    """Overall merged processing of histogram information."""
    t1 = histogram_tube(t1)
    t2 = histogram_tube(t2)
    x = layers.concatenate([t1, t2])
    x = layers.Dense(
        units=16, activation="elu",
        kernel_regularizer=regularizers.l2(GLOBAL_DECAY))(x)
    return x


def create_model_histo(xshape, yshape):
    """Create a simple sequential neural network with multiple inputs."""

    t1_input = layers.Input(shape=xshape[0])
    t2_input = layers.Input(shape=xshape[1])

    x = histogram_merged(t1_input, t2_input)
    final = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=[t1_input, t2_input], outputs=final)

    return model
