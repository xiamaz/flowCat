import tensorflow as tf


def calculate_bmu_indexes():
    """Create a tensorflow graph for calculating the best matched indexes in a SOM."""
    mapdata = tf.placeholder(tf.float32, shape=(None, None), name="som")
    fcsdata = tf.placeholder(tf.float32, shape=(None, None), name="fcs")
    squared_diffs = tf.pow(tf.subtract(
        tf.expand_dims(mapdata, axis=0),
        tf.expand_dims(fcsdata, axis=1)), 2)
    diffs = tf.reduce_sum(squared_diffs, 2)
    bmu = tf.argmin(diffs, axis=1)
    return bmu
