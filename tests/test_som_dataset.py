import unittest

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from flowcat.som_dataset import SOMDataset, SOMCase, SOMSequence

from . import shared


def create_som_case(id_label: str, group: str, data: dict) -> "SOMCase":
    np_dict = {tube: np.array(d) for tube, d in data.items()}
    return SOMCase(
        label=id_label,
        group=group,
        data=np_dict
    )


def create_som_dataset(data, config_data) -> "SOMDataset":
    data = pd.Series([create_som_case(*d) for d in data])
    config = {
        tube: {
            "dims": dims,
            "channels": channels,
        } for tube, dims, channels in config_data
    }
    return SOMDataset(data, config)


class SOMDatasetTestCase(shared.FlowcatTestCase):

    def test_simple_dataset(self):
        dataset = create_som_dataset(
            [
                ("1", "a", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("2", "a", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("3", "a", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("4", "a", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("5", "b", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("6", "b", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("7", "b", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("8", "b", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
            ],
            [
                ("1", (2, 2, 2), ("x", "y"))
            ]
        )

        with self.subTest("groups_counts"):
            self.assertEqual(dataset.group_counts, {"a": 4, "b": 4})

        with self.subTest("splitting"):
            ds_a, ds_b = dataset.split(0.75, stratified=True)
            self.assertEqual(ds_a.group_counts, {"a": 3, "b": 3})
            self.assertEqual(ds_b.group_counts, {"a": 1, "b": 1})

        with self.subTest("balance"):
            ds_bal = dataset.balance(10)
            self.assertEqual(ds_bal.group_counts, {"a": 10, "b": 10})

    def test_simple_sequence(self):
        dataset = create_som_dataset(
            [
                ("1", "a", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("2", "a", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("3", "a", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("4", "a", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("5", "b", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("6", "b", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("7", "b", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
                ("8", "b", {"1": [[[1, 1], [2, 1]], [[3, 1], [3, 2]]]}),
            ],
            [
                ("1", (2, 2, 2), ("x", "y"))
            ]
        )

        binarizer = LabelBinarizer()
        binarizer.fit(["a", "b", "c"])
        sequence = SOMSequence(dataset, binarizer, ["1"], batch_size=1)
        self.assertEqual(len(sequence), 8)

        expected = (np.array([
            [[[[1, 1], [2, 1]], [[3, 1], [3, 2]]]],
        ]), np.array([[1, 0, 0]]))
        for a, b in zip(sequence[0], expected):
            assert_array_equal(a, b)
