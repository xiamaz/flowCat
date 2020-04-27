import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from flowcat.types import fcsdata as fcs
from flowcat.preprocessing import scalers
from flowcat.types.marker import Marker


class MinMaxScalerTest(unittest.TestCase):

    def test_basic(self):
        fit_to_range = False

        test_datas = [
            (
                "test_a",
                fcs.FCSData(
                    (np.array([
                        [1, 2, 3, 10],
                        [4, 2, 5, 99]
                    ]).T, np.ones((4, 2))),
                    channels=["A", "B"],
                ),
                fcs.FCSData(
                    (np.array([
                        [0.0, 0.111111, 0.222222, 1.0],
                        [0.020619, 0.0, 0.030928, 1.0]
                    ]).T, np.ones((4, 2))),
                    channels=[
                        Marker.name_to_marker("A", fcs.ChannelMeta(0.0, 1.0, pne=(0, 0), png=0)),
                        Marker.name_to_marker("B", fcs.ChannelMeta(0.0, 1.0, pne=(0, 0), png=0))
                    ]
                ),
                False
            )
        ]

        for name, testdata, expected, fit_to_range in test_datas:
            with self.subTest(name=name, fit_to_range=fit_to_range):
                model = scalers.FCSMinMaxScaler(fit_to_range=fit_to_range)
                result = model.fit_transform(testdata)
                assert_array_almost_equal(result.data, expected.data)
                assert_array_almost_equal(result.ranges_array, expected.ranges_array)


class StandardScalerTest(unittest.TestCase):

    def test_basic(self):
        test_datas = [
            (
                "test_a",
                fcs.FCSData(
                    (np.array([
                        [1, 2, 3, 10],
                        [4, 2, 5, 99]
                    ]).T, np.ones((4, 2))),
                    channels=["A", "B"],
                ),
                fcs.FCSData(
                    (np.array([
                        [-0.84852815, -0.56568545, -0.28284273, 1.6970563],
                        [-0.56908065, -0.61751306, -0.5448645, 1.7314582]
                    ]).T, np.ones((4, 2))),
                    channels=[
                        Marker.name_to_marker("A", fcs.ChannelMeta(-0.84852815, 1.6970563, pne=(0, 0), png=0)),
                        Marker.name_to_marker("B", fcs.ChannelMeta(-0.61751306, 1.7314582, pne=(0, 0), png=0)),
                    ]
                ),
            )
        ]

        for name, testdata, expected, in test_datas:
            with self.subTest(name=name):
                model = scalers.FCSStandardScaler()
                result = model.fit_transform(testdata)
                assert_array_almost_equal(result.data, expected.data)
                assert_array_almost_equal(result.ranges_array, expected.ranges_array)
