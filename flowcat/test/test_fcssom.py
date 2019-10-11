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
            (3, 3, 2),
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
            [[0.8140727, 0.23507498],
             [0.55659986, 0.20142278],
             [0.2574054, 0.1715866 ]],
            [[0.7871901, 0.53189766],
             [0.4951395, 0.48251832],
             [0.19067009, 0.4425945 ]],
            [[0.7575845, 0.8054883 ],
             [0.45718592, 0.7829037 ],
             [0.19094288, 0.75963354]],
        ])
        model.train([traindata])
        result = model.transform(testdata)
        assert_array_almost_equal(result.data, expected)
