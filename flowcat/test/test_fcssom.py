import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from flowcat.dataset import fcs
from flowcat.sommodels import fcssom


SEED = 42

MARKERS = ["AA", "BB"]

np.random.seed(SEED)


class FCSSomTestCase(unittest.TestCase):

    def test_basic_train(self):
        model = fcssom.FCSSom(
            (2, 2, 2),
            seed=SEED,
            markers=MARKERS,
        )

        traindata = fcs.FCSData(
            (np.random.rand(1000, 2), np.ones((1000, 2))),
            channels=MARKERS,
        )
        testdata = fcs.FCSData(
            (np.random.rand(1000, 2), np.ones((1000, 2))),
            channels=MARKERS,
        )

        expected = np.array([
            [[0.29068232, 0.3177734], [0.6773688, 0.29500887]],
            [[0.3204862, 0.7031209], [0.69967717, 0.672693]],
        ])
        model.train([traindata])
        result = model.transform(testdata)
        assert_array_almost_equal(result.data, expected)
