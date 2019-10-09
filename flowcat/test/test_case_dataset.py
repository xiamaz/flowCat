"""
Case collection tests.
"""
import unittest
import json
import pathlib

import pandas as pd
from pandas.testing import assert_series_equal

from flowcat.dataset import case_dataset

from . import shared


class TestBasicCase(unittest.TestCase):
    """Basic case collection tests."""

    @classmethod
    def setUpClass(cls):
        cls.cases = case_dataset.CaseCollection.from_path(shared.FCS_PATH)

    def test_wrong_constructor(self):
        """Exception should be thrown if we try to instantiate the CaseCollection using a string or pathlike object."""
        self.assertRaises(ValueError, case_dataset.CaseCollection, "teststr")
        self.assertRaises(ValueError, case_dataset.CaseCollection, pathlib.Path("teststr"))
        self.assertRaises(ValueError, case_dataset.CaseCollection, shared.utils.URLPath("teststr"))

    def test_json_same(self):
        """Assert that exported json is identical to imported information."""

        with open(str(case_dataset.get_meta(shared.FCS_PATH, "case_info.json")), "r") as f:
            case_jsons = json.load(f)

        for cjson, case in zip(case_jsons, self.cases.data):
            with self.subTest(label=cjson["id"]):
                self.assertDictEqual(case.json, cjson)

    def test_group_num(self):
        """Test that the count function is working correctly."""
        self.assertDictEqual(self.cases.group_count, {"normal": 6, "CLL": 2, "LPL": 1, "PL": 1})

    def test_tubes(self):
        """Infer tubes from availability in input data."""
        self.assertListEqual(self.cases.tubes, [1, 2, 3])

    def test_markers(self):
        """Check that the returned markers match the given ones."""
        reference = pd.Series({
            "FS INT LIN": 1.0,
            "SS INT LIN": 1.0,
            "FMC7-FITC": 1.0,
            "CD10-PE": 1.0,
            "IgM-ECD": 1.0,
            "CD79b-PC5.5": 1.0,
            "CD20-PC7": 1.0,
            "CD23-APC": 1.0,
            "nix-APCA700": 1.0,
            "CD19-APCA750": 1.0,
            "CD5-PacBlue": 1.0,
            "CD45-KrOr": 1.0,
        })

        assert_series_equal(self.cases.get_markers(1), reference)

    def test_filter(self):
        """Check that filtering case cohorts works."""
        with self.subTest(i="tube filtering"):
            tubes_only = self.cases.filter(tubes=[1, 2])
            # didnt filter out any cases
            self.assertEqual(len(tubes_only), len(self.cases))
            self.assertListEqual(tubes_only.selected_tubes, [1, 2])
            self.assertDictEqual(tubes_only.selected_markers, {
                1: [
                    'FS INT LIN',
                    'SS INT LIN',
                    'FMC7-FITC',
                    'CD10-PE',
                    'IgM-ECD',
                    'CD79b-PC5.5',
                    'CD20-PC7',
                    'CD23-APC',
                    'CD19-APCA750',
                    'CD5-PacBlue',
                    'CD45-KrOr'],
                2: [
                    'FS INT LIN',
                    'SS INT LIN',
                    'Kappa-FITC',
                    'Lambda-PE',
                    'CD38-ECD',
                    'CD25-PC5.5',
                    'CD11c-PC7',
                    'CD103-APC',
                    'CD19-APCA750',
                    'CD22-PacBlue',
                    'CD45-KrOr']})

        with self.subTest(i="group filtering"):
            single_group = self.cases.filter(groups=["normal"])
            self.assertEqual(len(single_group), sum(g == "normal" for g in self.cases.groups))

        with self.subTest(i="num filtering"):
            one_each = self.cases.filter(num=1)
            self.assertEqual(len(one_each), len(set(self.cases.groups)))

        with self.subTest(i="infiltration filtering"):
            high_infil = self.cases.filter(infiltration=57)
            for case in high_infil:
                with self.subTest(i=case.id, group=case.group):
                    if case.group != "normal":
                        self.assertGreaterEqual(case.infiltration, 57)

        with self.subTest(i="count filtering"):
            high_count = self.cases.filter(counts=50000)
            for case in high_count:
                for tubesample in case.filepaths:
                    with self.subTest(i=case.id, p=tubesample.path):
                        self.assertGreaterEqual(tubesample.count, 50000)

    def test_view(self):
        sample_view = self.cases.filter(num=1)
        sample_tubeview = sample_view.get_tube(1)
        # all samples should be from the first tube
        for sample in sample_tubeview:
            with self.subTest(i=sample.parent.id):
                self.assertEqual(sample.tube, 1)
