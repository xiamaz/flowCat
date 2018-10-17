"""
Classifiers combining different inputs.
"""
from keras import layers, models


def create_model_maphisto(xshape, yshape):
    """Create model using both histogram and SOM map information."""
    m1input = layers.Input(shape=xshape[0])
    m2input = layers.Input(shape=xshape[1])
    t1input = layers.Input(shape=xshape[2])
    t2input = layers.Input(shape=xshape[3])

    mm = sommap_merged(m1input, m2input)
    hm = histogram_merged(t1input, t2input)
    x = layers.concatenate([mm, hm])
    x = layers.Dense(32)(x)
    final = layers.Dense(yshape, activation="softmax")(x)

    model = models.Model(
        inputs=[m1input, m2input, t1input, t2input], outputs=final)
    return model


def create_model_mapfcs(xshape, yshape):
    """Create model combining fcs processing and map cnn."""
    fcsinput = layers.Input(shape=xshape[0])
    m1input = layers.Input(shape=xshape[1])
    m2input = layers.Input(shape=xshape[2])

    fm = fcs_merged(fcsinput)
    mm = sommap_merged(m1input, m2input)
    x = layers.concatenate([fm, mm])
    final = layers.Dense(yshape, activation="softmax")(x)

    model = models.Model(inputs=[fcsinput, m1input, m2input], outputs=final)
    return model


def create_model_all(xshape, yshape):
    """Create model combining fcs, histogram and sommap information."""
    fcsinput = layers.Input(shape=xshape[0])
    m1input = layers.Input(shape=xshape[1])
    m2input = layers.Input(shape=xshape[2])
    t1input = layers.Input(shape=xshape[3])
    t2input = layers.Input(shape=xshape[4])

    fm = fcs_merged(fcsinput)
    mm = sommap_merged(m1input, m2input)
    hm = histogram_merged(t1input, t2input)
    x = layers.concatenate([fm, mm, hm])
    x = layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l=GLOBAL_DECAY))(x)
    final = layers.Dense(yshape, activation="softmax")(x)

    model = models.Model(
        inputs=[fcsinput, m1input, m2input, t1input, t2input], outputs=final)
    return model