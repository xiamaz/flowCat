import logging
import unittest

import numpy as np
from numpy.testing import assert_array_equal

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

        model.train(data)

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
        mapped, = model.run_till_op("BMU_Indices/map_to_node_index", newdata, 0)
        assert_array_equal(mapped, expected)

    def test_missing_data(self):
        model = tfsom.TFSom(
            (10, 10, 4),
            seed=SEED,
        )
        model.initialize()

        data = np.random.rand(1000, 4)
        model.train(data)

        newdata = np.random.rand(100, 4)
        newdata[:, 0] = 0
        newdata[:, 2] = 0
        mask = (0, 1, 0, 1)
        print(newdata)
        result = model.run_till_op("BMU_Indices/map_to_node_index", newdata, 0)


logging.basicConfig(level=logging.INFO)
