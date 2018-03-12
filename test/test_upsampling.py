'''
Test UpsamplingData which is used for loading and manipulating csv data
before actual classification.
'''
import unittest
from lib import upsampling

class TestUpsamplingData(unittest.TestCase):

    test_data = [("test/data/test_tube1.csv", "test/data/test_tube2.csv")]

    def setUp(self):
        self.upsampling = upsampling.UpsamplingData.from_files(
            self.test_data)

    def test_group_separation(self):
        data_groups = list(self.upsampling._data.groupby("group"))
        internal_groups = self.upsampling._groups
        group_asserts = {}
        for gname, gdata in data_groups:
            group_asserts[gname] = False
            for iname, idata in internal_groups:
                if gname == iname and gdata.shape == idata.shape:
                    group_asserts[gname] = True
        self.assertTrue(all(group_asserts.values()),
                        "Generated groups not identical or groups missing.")

    def test_group_selection(self):

        test_groups = self.upsampling.group_names[0:3]
        self.upsampling.select_groups(test_groups)
        self.assertEqual(test_groups, self.upsampling.group_names,
                         "Group names are not equal after selection.")

    def test_select_small_cohorts(self):
        small_cutoff = 80
        self.upsampling.exclude_small_cohorts(small_cutoff)
        group_sizes = [g.shape[0] for _, g in self.upsampling._groups]
        self.assertTrue(all([s >= small_cutoff for s in group_sizes]),
                        "Group sizes not smaller than cutoff after limiting.")

    def test_test_train_split(self):
        # test ratio splitting
        train, test = self.upsampling.get_test_train_split(ratio=0.8)
        self.assertAlmostEqual(
            0.8, train.shape[0] / (train.shape[0] + test.shape[0]),
            places=1,
            msg="Ratio for ratio split is inaccurate.")

        self.assertEqual(
            self.upsampling._data.shape[0],
            train.shape[0] + test.shape[0],
            "Ratio split not having right dimensions.")

        # test abs splitting
        abs_num = 10
        self.upsampling.exclude_small_cohorts(abs_num)
        train, test = self.upsampling.get_test_train_split(abs_num=abs_num)
        self.assertEqual(
            len(self.upsampling.group_names) * 10,
            test.shape[0], "Absolute split is wrong.")

        self.assertEqual(
            self.upsampling._data.shape[0],
            train.shape[0] + test.shape[0],
            "Absolute split not having right dimensions.")

    def test_k_fold_split(self):
        k_num = 5
        k_splits = self.upsampling.k_fold_split(k_num=k_num)
        self.assertEqual(k_num, len(k_splits), "Not correct number of splits")
        self.assertEqual(
            self.upsampling._data.shape[0],
            sum([s.shape[0] for s in k_splits]),
            "Case numbers incorrect in split.")
