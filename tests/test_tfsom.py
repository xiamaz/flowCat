import logging
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from flowcat.sommodels import tfsom

from .shared import TESTPATH


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
            79, 64, 92, 95, 39, 41, 39, 86, 35,
            7, 70, 55, 50, 88, 60, 10, 90,
            18, 15, 86, 51, 63, 96, 37, 75, 87,
            79, 98, 20, 90, 23, 6, 74, 88,
            70, 53, 29, 39, 93, 5, 85, 83,
            53, 4, 38, 78, 45, 5, 78, 33, 90,
            11, 64, 51, 1, 13, 17, 30, 48, 68,
            97, 48, 59, 45, 91, 58, 87, 6, 12,
            73, 67, 15, 59, 59, 31, 3, 70, 8, 69,
            71, 16, 9, 57, 77, 59, 30, 40, 6, 14,
            68, 2, 56, 64, 11, 37, 20, 98, 64, 40, 74
        ])

        newdata = np.random.rand(100, 4)
        newmask = np.ones((100, 4))
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
