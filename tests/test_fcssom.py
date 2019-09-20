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
            [[0.23379141, 0.81184906],
             [0.20201758, 0.52560365],
             [0.18959665, 0.2370348]],
            [[0.5282842, 0.7789625],
             [0.5025789, 0.47923133],
             [0.4661186, 0.19752763]],
            [[0.7997755, 0.7657486],
             [0.792592, 0.4714995],
             [0.763365, 0.18549682]]
        ])
        model.train([traindata])
        result = model.transform(testdata)
        assert_array_almost_equal(result.data, expected)
