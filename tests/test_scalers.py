import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from flowcat.dataset import fcs
from flowcat.preprocessing import scalers

from .shared import TESTPATH


class MinMaxScalerTest(unittest.TestCase):

    def test_basic(self):
        fit_to_range = False

        test_datas = [
            (
                "test_a",
                fcs.FCSData(
                    pd.DataFrame({"A": [1, 2, 3, 10], "B": [4, 2, 5, 99]}),
                ),
                fcs.FCSData(
                    pd.DataFrame({
                        "A": [0.0, 0.111111, 0.222222, 1.0],
                        "B": [0.020619, 0.0, 0.030928, 1.0]
                    }),
                    {
                        "A": fcs.ChannelMeta(pd.Interval(0.0, 1.0, "both"), True),
                        "B": fcs.ChannelMeta(pd.Interval(0.0, 1.0, "both"), False)
                    }
                ),
                False
            )
        ]

        for name, testdata, expected, fit_to_range in test_datas:
            with self.subTest(name=name, fit_to_range=fit_to_range):
                model = scalers.FCSMinMaxScaler(fit_to_range=fit_to_range)
                result = model.fit_transform(testdata)
                assert_frame_equal(result.data, expected.data, check_less_precise=True)
                assert_frame_equal(result.ranges_dataframe, expected.ranges_dataframe, check_less_precise=True)


class StandardScalerTest(unittest.TestCase):

    def test_basic(self):
        fit_to_range = False

        test_datas = [
            (
                "test_a",
                fcs.FCSData(
                    pd.DataFrame({"A": [1, 2, 3, 10], "B": [4, 2, 5, 99]}),
                ),
                fcs.FCSData(
                    pd.DataFrame({
                        "A": [-0.84852815, -0.56568545, -0.28284273, 1.6970563],
                        "B": [-0.56908065, -0.61751306, -0.5448645, 1.7314582]
                    }),
                    {
                        "A": fcs.ChannelMeta(pd.Interval(-0.84852815, 1.6970563, "both"), True),
                        "B": fcs.ChannelMeta(pd.Interval(-0.61751306, 1.7314582, "both"), False)
                    }
                ),
            )
        ]

        for name, testdata, expected, in test_datas:
            with self.subTest(name=name):
                model = scalers.FCSStandardScaler()
                result = model.fit_transform(testdata)
                assert_frame_equal(result.data, expected.data, check_less_precise=True)
                assert_frame_equal(result.ranges_dataframe, expected.ranges_dataframe, check_less_precise=True)
