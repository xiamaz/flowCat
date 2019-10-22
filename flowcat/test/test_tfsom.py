import logging
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from flowcat.sommodels import tfsom

SEED = 42


np.random.seed(SEED)


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
            69, 49, 21, 28, 59, 35, 46, 84, 40, 13
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


logging.basicConfig(level=logging.INFO)
