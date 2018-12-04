import unittest
import collections
import pathlib

from .shared import *

from flowcat.data import loaders
from flowcat.data import all_dataset
from tests.test_dataset import TestCombinedDataset


class TestLoaders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = all_dataset.CombinedDataset.from_paths(
            casepath=FCS_PATH, paths=[("FCS", FCS_PATH), ("HISTO", HISTO_PATH), ("SOM", SOM_PATH)],
            group_names=["CLL", "LPL", "PL", "normal"],
        )
        cls.output_spec = [
            loaders.loader_builder(
                loaders.FCSLoader.create_inferred, tubes=[1, 2]
            ),
            loaders.loader_builder(
                loaders.CountLoader.create_inferred, tube=1,
                datatype="dataframe"
            ),
            loaders.loader_builder(
                loaders.CountLoader.create_inferred, tube=2,
                datatype="dataframe"
            ),
            loaders.loader_builder(
                loaders.Map2DLoader.create_inferred, tube=1,
                sel_count="counts",
            ),
            loaders.loader_builder(
                loaders.Map2DLoader.create_inferred, tube=2,
                pad_width=1,
            ),
        ]
        cls.seq = loaders.DatasetSequence(cls.data, cls.output_spec, draw_method="sequential")

        cls.xshapes = [
            (400, 36),
            (100, ),
            (100, ),
            (32, 32, 12, ),
            (34, 34, 11, ),
        ]
        cls.yshape = (4, )

    def get_batch_shape(self, batch_size):
        return [[(batch_size, *t) for t in self.xshapes], (batch_size, * self.yshape)]

    def assertShape(self, data, batch_size):
        """Ensure that output data matches our input specification."""
        xdata, ydata = data
        dshapes = [d.shape for d in xdata]
        shapes = [dshapes, ydata.shape]
        self.assertEqual(shapes, self.get_batch_shape(batch_size))

    def test_labels(self):
        """Check that the returned label list is correct."""
        self.assertListEqual(
            list(zip(self.seq.labels, self.seq.randnums, self.seq.ylabels)), self.seq.label_groups)

    def test_shapes(self):
        """Check that the returned shape of the sequantial object is correct."""
        self.assertEqual(
            self.seq.shape,
            ([(400, 36), (100, ), (100, ), (32, 32, 12), (34, 34, 11)], 4)
        )

    def test_num_batches(self):
        """Check that the number of calculated batches is correct"""
        self.assertEqual(len(self.seq), 1)

    def test_get_index(self):
        """Check that getting a single batch will return data of the correct shape."""
        self.assertShape(self.seq[0], batch_size=10)

    def test_draw_methods(self):
        """Check that the different draw methods behave as expected."""
        seq_truth = self.data.label_groups
        labels = set(l for l, _ in seq_truth)
        with self.subTest("sequential"):
            seq_ds = [(l, g) for l, _, g in self.seq.label_groups]
            self.assertListEqual(seq_ds, seq_truth)
        with self.subTest("shuffle"):
            shuffled = loaders.DatasetSequence(self.data, self.output_spec, draw_method="shuffle")
            shuffled_labels = set(l for l, _, _ in shuffled.label_groups)
            self.assertSetEqual(shuffled_labels, labels)
        with self.subTest("balanced_nodict"):
            equal_truth = {
                "CLL": 20,
                "LPL": 20,
                "PL": 20,
                "normal": 20,
            }
            equalized = loaders.DatasetSequence(self.data, self.output_spec, draw_method="balanced", epoch_size=80)
            equal_labels = equalized.label_groups
            equal_counts = collections.Counter(g for _, _, g in equal_labels)
            self.assertDictEqual(equal_counts, equal_truth)
        with self.subTest("balanced_correctdict"):
            group_nums = {
                "CLL": 10,
                "LPL": 20,
                "PL": 5,
                "normal": 5
            }
            grouped = loaders.DatasetSequence(
                self.data, self.output_spec, draw_method="balanced", number_per_group=group_nums)
            grouped_labels = equalized.label_groups
            grouped_counts = collections.Counter(g for _, _, g in grouped_labels)
            self.assertDictEqual(grouped_counts, grouped_counts)
        with self.subTest("balanced_wrongdict"):
            incomplete_nums = {
                "CLL": 100,
            }
            with self.assertRaises(AssertionError):
                incomplete_grouped = loaders.DatasetSequence(
                    self.data, self.output_spec, draw_method="balanced", number_per_group=incomplete_nums)

    def test_generate_batches(self):
        """Test generation of samples using multiprocessing."""
        shuffled = loaders.DatasetSequence(self.data, self.output_spec, draw_method="shuffle", batch_size=2)
        num_workers = [1, 4, 8]
        for num_worker in num_workers:
            with self.subTest(num_workers=num_workers):
                shuffled.generate_batches(num_workers=num_worker)
                res = shuffled._cache
                for r in res:
                    self.assertShape(r, batch_size=2)
                self.assertEqual(len(res), len(shuffled))
                self.assertShape(shuffled[0], batch_size=2)
