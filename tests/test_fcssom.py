import logging
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from flowcat.dataset import fcs
from flowcat.sommodels import fcssom


SEED = 42

MARKERS = ("AA", "BB")

np.random.seed(SEED)


class FCSSomTestCase(unittest.TestCase):

    def test_basic_train(self):
        model = fcssom.FCSSom(
            (4, 4, 2),
            seed=SEED,
            markers=MARKERS,
        )

        traindata = fcs.FCSData(
            pd.DataFrame(np.random.rand(1000, 2), columns=MARKERS)
        )
        testdata = fcs.FCSData(
            pd.DataFrame(np.random.rand(1000, 2), columns=MARKERS)
        )

        model.train([traindata])
        result = model.transform(testdata)
        print(result.data)
