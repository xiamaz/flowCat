import logging
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import tensorflow as tf

from flowcat.sommodels import tfsom
from flowcat.utils.time_timers import timer

SEED = 42


np.random.seed(SEED)


def quantization_error_model():
    mapdata = tf.placeholder(tf.float32, shape=(None, None), name="som")
    fcsdata = tf.placeholder(tf.float32, shape=(None, None), name="fcs")
    squared_diffs = tf.pow(tf.subtract(
        tf.expand_dims(mapdata, axis=0),
        tf.expand_dims(fcsdata, axis=1)), 2)
    diffs = tf.reduce_sum(squared_diffs, 2)
    euc_distance = tf.sqrt(tf.reduce_min(diffs, axis=1))
    qe = tf.reduce_mean(euc_distance)
    return qe


class TFSomTestCase(unittest.TestCase):

    def test_basic_train(self):
        model = tfsom.TFSom(
            (10, 10, 4),
            seed=SEED,
        )

        model.initialize()

        data = np.random.rand(1000, 4)
        mask = np.ones((1000, 4))

        model.train(data, mask)

        expected = np.array([
            55, 47, 8, 28, 7, 52, 26, 15, 96, 8
        ])

        newdata = np.random.rand(10, 4)
        newmask = np.ones((10, 4))
        mapped, = model.run_till_op("BMU_Indices/map_to_node_index", newdata, newmask, 0)
        assert_array_equal(mapped, expected)

    def test_small_batch(self):
        """Check that different batch sizes do not affect training results."""
        data = np.random.rand(1000, 4)
        mask = np.ones((1000, 4))

        model = tfsom.TFSom(
            (10, 10, 4),
            seed=SEED,
            batch_size=1000,
        )
        model.initialize()
        model.train(data, mask)
        single_batch = model.output_weights

        model = tfsom.TFSom(
            (10, 10, 4),
            seed=SEED,
            batch_size=100,
        )
        model.initialize()
        model.train(data, mask)
        multi_batch = model.output_weights
        assert_allclose(single_batch, multi_batch, rtol=1e-04)

    def test_missing_data(self):
        model = tfsom.TFSom(
            (3, 3, 4),
            seed=SEED,
        )
        model.initialize()

        data = np.random.rand(5, 4)
        mask = np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ])
        model.train(data, mask)
        result = model.output_weights[:, -1]
        # nulled channel should also have nulled weights, since we are
        # using batched training algorithm
        assert_array_equal(result, np.zeros(result.shape))

    def test_quantization_error(self):
        """Performance test, asserting that quantization performance has not degraded significantly."""
        sess = tf.Session()
        qe_model = quantization_error_model()
        model = tfsom.TFSom((32, 32, 10), seed=SEED, max_epochs=20, batch_size=50000).initialize()
        traindata = np.random.rand(50000, 10)
        trainmask = np.ones((50000, 10))
        data = np.random.rand(100, 10)
        mask = np.ones((100, 10))

        with timer("Training time"):
            model.train(traindata, trainmask)

        transformed = model.transform(data, mask)
        error = sess.run(qe_model, feed_dict={"fcs:0": data, "som:0": transformed})
        self.assertTrue(error <= 0.1)


logging.basicConfig(level=logging.INFO)
