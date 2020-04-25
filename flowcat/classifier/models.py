from keras import layers, regularizers, models


def create_model_multi_input(input_shapes, yshape, global_decay=5e-6) -> models.Model:
    """Create a multi input model for keras."""
    segments = []
    inputs = []
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
        #     filters=48, kernel_size=2, activation="relu", strides=1,
        #     kernel_regularizer=regularizers.l2(global_decay),
        # )(x)
        x = layers.Conv2D(
            filters=64, kernel_size=2, activation="relu", strides=2,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

        # x = layers.GlobalAveragePooling2D()(x)
        x = layers.GlobalMaxPooling2D()(x)
        segments.append(x)

    if len(segments) > 1:
        x = layers.concatenate(segments)
    else:
        x = segments[0]

    x = layers.Dense(
        units=64, activation="relu",
        # kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    x = layers.Dense(
        units=32, activation="relu",
        # kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)

    x = layers.Dense(
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model
