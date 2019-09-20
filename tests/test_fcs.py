"""Test processing of FCS files."""
import unittest

import numpy as np
from numpy.testing import assert_array_equal
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
            assert_array_equal(first.data, second.data)

        with self.subTest("from np array"):
            data = np.random.rand(10, 4)
            mask = np.ones((10, 4))
            channels = ["a", "b", "c", "d"]
            fcs.FCSData((data, mask), channels=channels)

    def test_rename(self):
        testdata = fcs.FCSData(
            (np.zeros((10, 4)), np.zeros((10, 4))),
            channels=["a", "b", "c", "d"]
        )
        testdata.rename({"a": "z"})
        self.assertEqual(testdata.channels, ["z", "b", "c", "d"])
        self.assertIn("z", testdata.meta)
        self.assertNotIn("a", testdata.meta)

    def test_marker_to_name_only(self):
        testdata = fcs.FCSData(
            (np.zeros((10, 4)), np.zeros((10, 4))),
            channels=["a-a", "b-b", "c-c", "d-d"]
        )
        testdata.marker_to_name_only()
        self.assertEqual(testdata.channels, ["a", "b", "c", "d"])

    def test_reorder_channels(self):
        data = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [7, 8, 9],
        ])
        testdata = fcs.FCSData(
            (data, np.ones(data.shape)),
            channels=["A", "B", "C"]
        )
        testdata.reorder_channels(["B", "A", "C"])
        self.assertEqual(testdata.channels, ["B", "A", "C"])
        assert_array_equal(testdata.data, np.array([
            [1, 0, 2],
            [1, 0, 2],
            [8, 7, 9],
        ]))
        testdata.reorder_channels(["C", "B", "A"])
        self.assertEqual(testdata.channels, ["C", "B", "A"])
        assert_array_equal(testdata.data, np.array([
            [2, 1, 0],
            [2, 1, 0],
            [9, 8, 7],
        ]))

    def test_add_missing(self):
        data = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [7, 8, 9],
        ])
        testdata = fcs.FCSData(
            (data, np.ones(data.shape)),
            channels=["A", "B", "C"]
        )
        testdata.add_missing_channels(["D"])
        self.assertEqual(testdata.channels, ["A", "B", "C", "D"])
        assert_array_equal(testdata.data, np.array([
            [0, 1, 2, 0],
            [0, 1, 2, 0],
            [7, 8, 9, 0],
        ]))
        assert_array_equal(testdata.mask, np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ]))

        with self.assertRaises(ValueError):
            testdata.add_missing_channels(["A"])

    def test_drop_channels(self):
        data = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [7, 8, 9],
        ])
        testdata = fcs.FCSData(
            (data, np.ones(data.shape)),
            channels=["A", "B", "C"]
        )
        testdata.drop_channels(["A"])
        self.assertEqual(testdata.channels, ["B", "C"])
        self.assertNotIn("A", testdata.meta)
        assert_array_equal(testdata.data, np.array([
            [1, 2],
            [1, 2],
            [8, 9],
        ]))

    def test_align(self):
        data = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [7, 8, 9],
        ])
        testdata = fcs.FCSData(
            (data, np.ones(data.shape)),
            channels=["A", "B", "C"]
        )
        aligned = testdata.align(["B", "D", "A"])
        self.assertEqual(aligned.channels, ["B", "D", "A"])
        assert_array_equal(aligned.data, np.array([
            [1, 0, 0],
            [1, 0, 0],
            [8, 0, 7],
        ]))
        assert_array_equal(aligned.mask, np.array([
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
        ]))

    def test_join(self):
        with self.subTest("diff_same_channels"):
            first = fcs.FCSData(shared.DATAPATH / "fcs/cll1_1.lmd")
            second = fcs.FCSData(shared.DATAPATH / "fcs/cll2_1.lmd")
            joined = fcs.join_fcs_data([first, second])
            self.assertEqual(joined.shape, (100000, 12))
            self.assertEqual(set(joined.channels), {
                'FMC7-FITC',
                'CD23-APC',
                'FS INT LIN',
                'CD10-PE',
                'CD5-PacBlue',
                'SS INT LIN',
                'CD20-PC7',
                'CD45-KrOr',
                'CD79b-PC5.5',
                'nix-APCA700',
                'CD19-APCA750',
                'IgM-ECD'
            })

        with self.subTest("same_case"):
            first = fcs.FCSData(shared.DATAPATH / "fcs/cll1_1.lmd")
            second = fcs.FCSData(shared.DATAPATH / "fcs/cll1_2.lmd")
            joined = fcs.join_fcs_data([first, second])
            self.assertEqual(joined.shape, (100000, 19))
            self.assertEqual(max(np.sum(joined.mask, axis=0)), 100000)
