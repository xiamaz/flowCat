import unittest

import pandas as pd
from flowcat.som_dataset import SOMDataset, SOMCase


class SOMDatasetTestCase(unittest.TestCase):

    def test_simple(self):
        data = pd.Series([
            SOMCase("1", "a", {}, {}),
            SOMCase("2", "a", {}, {}),
            SOMCase("3", "a", {}, {}),
            SOMCase("4", "a", {}, {}),
            SOMCase("5", "b", {}, {}),
            SOMCase("6", "b", {}, {}),
            SOMCase("7", "b", {}, {}),
            SOMCase("8", "b", {}, {}),
        ])
        dataset = SOMDataset(data, {})
        with self.subTest("groups_counts"):
            self.assertEqual(dataset.group_counts, {"a": 4, "b": 4})

        with self.subTest("splitting"):
            ds_a, ds_b = dataset.split(0.75, stratified=True)
            self.assertEqual(ds_a.group_counts, {"a": 3, "b": 3})
            self.assertEqual(ds_b.group_counts, {"a": 1, "b": 1})

        with self.subTest("balance"):
            ds_bal = dataset.balance(10)
            self.assertEqual(ds_bal.group_counts, {"a": 10, "b": 10})
