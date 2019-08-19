"""Test processing of FCS files."""
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
import fcsparser

from flowcat.dataset import fcs

from . import shared


class TestFCS(unittest.TestCase):

    def test_simple(self):
        """Test simple creation of FCS data objects."""
        with self.subTest("from path"):
            fcs.FCSData(shared.DATAPATH / "fcs/cll1_1.lmd")

        with self.subTest("from another object"):
            first = fcs.FCSData(shared.DATAPATH / "fcs/cll1_1.lmd")
            second = fcs.FCSData(first)
            assert_frame_equal(first.data, second.data)

        with self.subTest("from meta data tuple"):
            meta, data = fcsparser.parse(str(shared.DATAPATH / "fcs/cll1_1.lmd"))
            created = fcs.FCSData(data, meta=meta)
            assert_frame_equal(created.data, data, check_dtype=False)

    def test_join(self):
        first = fcs.FCSData(shared.DATAPATH / "fcs/cll1_1.lmd")
        second = fcs.FCSData(shared.DATAPATH / "fcs/cll2_1.lmd")
        joined = fcs.join_fcs_data([first, second])
        print(joined)

    # def test_ranges(self):
    #     """Check whether range information is actually valid."""
    #     meta = {
    #         "$P1R": 10,
    #         "$P2R": 20,
    #     }
    #     data = pd.DataFrame({"Col1": [1, 2, 3, 4], "Col2": [4, 5, 6, 7]})
    #     tfcs = fcs.FCSData((meta, data))
    #     expected = pd.DataFrame({
    #         "Col1": {"min": 0.0, "max": 10.0},
    #         "Col2": {"min": 0.0, "max": 20.0},
    #     })
    #     assert_frame_equal(expected, tfcs.ranges)
