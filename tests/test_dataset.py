import unittest
import pathlib

from .shared import *

from flowcat.dataset import combined_dataset
from flowcat.mappings import GROUP_MAPS


def invert_dict(d):
    new = {}
    for k, v in d.items():
        if v not in new:
            new[v] = [k]
        else:
            new[v].append(k)
    return new


class TestCombinedDataset(unittest.TestCase):

    def setUp(self):
        self.data = combined_dataset.CombinedDataset.from_paths(
            paths={"FCS": FCS_PATH, "HISTO": HISTO_PATH, "SOM": SOM_PATH},
            group_names=["CLL", "LPL", "PL", "normal"],
        )

    def test_get_randnums(self):
        corr_randnums = {
            'a027a84b1b80e1f10666f9915ec8d5cb147fe2ac': [0],
            '36a99bbc331dd093fa5f7a25fe795f2e813a1a89': [0],
            '70d3d6cc1d4bbd6f942164fccdfeff7ba2cf685d': [0],
            '6cc25ce9abfd4d3fce14f4baee1c1ea48d7fd524': [0],
            '6dc13f6237012b484f2a50793ac0eafd0dd8aebf': [0],
            'eef5b3d2f21e780715a806532c7da42a2ed8cd3a': [0],
            '7c80bee8bc439994bdfdae58c3182a6d0bd29059': [0],
            '7b463e7431b89c59c7400749de627b6133184521': [0],
            'caae2a7dac944bddfc87b981c51e4be5ec3cdae6': [0],
            'fd50c85858e42771ad27af9ab0fb9d5cdc6c0d07': [0]
        }
        for dtype in ["FCS", "HISTO", "SOM"]:
            randnums = self.data.get_randnums(dtype)
            self.assertDictEqual(randnums, corr_randnums)

    def test_get_label_rand_group(self):
        corr_labels = [
            ('a027a84b1b80e1f10666f9915ec8d5cb147fe2ac', 0, 'LPL'),
            ('36a99bbc331dd093fa5f7a25fe795f2e813a1a89', 0, 'normal'),
            ('70d3d6cc1d4bbd6f942164fccdfeff7ba2cf685d', 0, 'normal'),
            ('6cc25ce9abfd4d3fce14f4baee1c1ea48d7fd524', 0, 'CLL'),
            ('6dc13f6237012b484f2a50793ac0eafd0dd8aebf', 0, 'PL'),
            ('eef5b3d2f21e780715a806532c7da42a2ed8cd3a', 0, 'normal'),
            ('7c80bee8bc439994bdfdae58c3182a6d0bd29059', 0, 'CLL'),
            ('7b463e7431b89c59c7400749de627b6133184521', 0, 'normal'),
            ('caae2a7dac944bddfc87b981c51e4be5ec3cdae6', 0, 'normal'),
            ('fd50c85858e42771ad27af9ab0fb9d5cdc6c0d07', 0, 'normal')
        ]
        labels = self.data.get_label_rand_group(["SOM"])
        self.assertListEqual(labels, corr_labels)

    def test_filter(self):
        data = self.data.filter(num=1, tubes=[1, 2])
        self.assertEqual(len(data.labels), 4)

    def test_availability(self):
        data = self.data.set_available(["FCS", "HISTO", "SOM"])
        self.assertEqual(len(data.labels), 10)

    def test_filter_available(self):
        data = self.data.filter(num=1, tubes=[1, 2])
        data.set_available(["FCS", "SOM"])
        self.assertEqual(len(data.labels), 4)

    def test_get(self):
        tubes = [1, 2]
        dtypes = ["FCS", "HISTO", "SOM"]
        self.data.filter(tubes=tubes)
        self.data.set_available(dtypes)
        for testlabel in self.data.labels:
            for dtype in dtypes:
                with self.subTest(dtype=dtype):
                    res = self.data.get(testlabel, dtype)
                    self.assertTrue(
                        all(pathlib.Path(res[n]).exists() for n in tubes))

    def test_mapping(self):
        data = self.data.set_mapping(GROUP_MAPS["3class"])
        self.assertListEqual(
            data.groups,
            [
                'CD5-', 'normal', 'normal', 'CD5+', 'CD5+',
                'normal', 'CD5+', 'normal', 'normal', 'normal'
            ]
        )

    def test_split_dataset(self):

        def valid_split(data, dtrain, dtest, ratio):
            """Check whether a given split is valid.
            This includes, whether the given ratio has been applied correctly.
            Ratios are always applied per group. The ceiling of the cutoff is used
            in separation, so that more cases will always end up in the training
            dataset.
            """
            train_labels = invert_dict(dict(zip(dtrain.labels, dtrain.groups)))
            test_labels = invert_dict(dict(zip(dtest.labels, dtest.groups)))

            for group in train_labels:
                train = train_labels.get(group, [])
                test = test_labels.get(group, [])
                with self.subTest(group):
                    if ratio >= 1:
                        grouplen = sum(1 for g in data.groups if g == group)
                        self.assertEqual(len(train), min(grouplen, ratio))
                    else:
                        cratio = len(train) / (len(train) + len(test))
                        if (len(train) + len(test)) > 2:
                            self.assertAlmostEqual(cratio, ratio, delta=0.1)
                        elif (len(train) + len(test)) > 1:
                            self.assertAlmostEqual(cratio, 0.5 if ratio < 0.75 else 1)
                        else:
                            self.assertTrue(train and not test, "Group with one sample is not located in the train set.")

            # check all labels are not duplicated
            self.assertEqual(len(data.labels), len(dtrain.labels + dtest.labels))
            self.assertEqual(len(data.labels), len(set(dtrain.labels + dtest.labels)))


        test_splits = [0.5, 0.75, 0.8, 0.9, 1]
        for split in test_splits:
            with self.subTest(split=split):
                train, test = combined_dataset.split_dataset(self.data, train_num=split)
                valid_split(self.data, train, test, split)

    def test_split_dataset_too_large(self):
        """Check that we return a value error if we try to use an absolute
        train number, which is larger than the cohort."""
        with self.assertRaises(ValueError):
            combined_dataset.split_dataset(self.data, train_num=1000)


    def test_split_dataset_seed(self):
        """Check that using a seed will always return the same split of the
        input data."""
        with self.subTest("Same seed returns same"):
            tr1, t1 = combined_dataset.split_dataset(self.data, train_num=0.5, seed=42)
            train = [
                'a027a84b1b80e1f10666f9915ec8d5cb147fe2ac',
                '70d3d6cc1d4bbd6f942164fccdfeff7ba2cf685d',
                '6dc13f6237012b484f2a50793ac0eafd0dd8aebf',
                'eef5b3d2f21e780715a806532c7da42a2ed8cd3a',
                '7c80bee8bc439994bdfdae58c3182a6d0bd29059',
                '7b463e7431b89c59c7400749de627b6133184521'
            ]
            test = [
                '36a99bbc331dd093fa5f7a25fe795f2e813a1a89',
                '6cc25ce9abfd4d3fce14f4baee1c1ea48d7fd524',
                'caae2a7dac944bddfc87b981c51e4be5ec3cdae6',
                'fd50c85858e42771ad27af9ab0fb9d5cdc6c0d07'
            ]
            self.assertListEqual(tr1.labels, train)
            self.assertListEqual(t1.labels, test)


        with self.subTest("Different seeds will return different splits on the same data."):
            tr1, t1 = combined_dataset.split_dataset(self.data, train_num=0.5, seed=42)
            tr2, t2 = combined_dataset.split_dataset(self.data, train_num=0.5, seed=24)
            self.assertTrue(any(t not in tr2.labels for t in tr1.labels))
            self.assertTrue(any(t not in t2.labels for t in t1.labels))
