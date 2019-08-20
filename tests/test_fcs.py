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
            meta = fcs.create_meta_from_fcs(meta, data)
            created = fcs.FCSData(data, meta=meta)
            assert_frame_equal(created.data, data, check_dtype=False)

    def test_join(self):
        first = fcs.FCSData(shared.DATAPATH / "fcs/cll1_1.lmd")
        second = fcs.FCSData(shared.DATAPATH / "fcs/cll2_1.lmd")
        joined = fcs.join_fcs_data([first, second])
        self.assertEqual(joined.shape, (100000, 12))

    def test_ranges_df(self):
        first = fcs.FCSData(shared.DATAPATH / "fcs/cll1_1.lmd")
        expected = pd.DataFrame({
            'FS INT LIN': {"max": 1024, "min": 0},
            'SS INT LIN': {"max": 1024, "min": 0},
            'FMC7-FITC': {"max": 1024, "min": 0},
            'CD10-PE': {"max": 1024, "min": 0},
            'IgM-ECD': {"max": 1024, "min": 0},
            'CD79b-PC5.5': {"max": 1024, "min": 0},
            'CD20-PC7': {"max": 1024, "min": 0},
            'CD23-APC': {"max": 1024, "min": 0},
            'nix-APCA700': {"max": 1024, "min": 0},
            'CD19-APCA750': {"max": 1024, "min": 0},
            'CD5-PacBlue': {"max": 1024, "min": 0},
            'CD45-KrOr': {"max": 1024, "min": 0},
        })
        assert_frame_equal(first.ranges_dataframe, expected)
