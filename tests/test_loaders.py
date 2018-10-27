import unittest
import pathlib

from flowcat.data import loaders
from flowcat.data import all_dataset
from tests.test_dataset import TestCombinedDataset

TESTPATH = pathlib.Path(__file__).parent
DATAPATH = TESTPATH / "data"

FCS_PATH = DATAPATH / "small_dataset"
SOM_PATH = DATAPATH / "som" / "testrun_s32_ttoroid"
HISTO_PATH = DATAPATH / "histogram" / "abstract_normal"


class TestLoaders(unittest.TestCase):

    def setUp(self):
        self.data = all_dataset.CombinedDataset.from_paths(
            casepath=FCS_PATH, paths=[("HISTO", HISTO_PATH), ("SOM", SOM_PATH)]
        )
        output_spec = [
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
        self.seq = loaders.DatasetSequence(self.data, output_spec)

    def test_labels(self):
        self.assertListEqual(
            list(zip(self.seq.labels, self.seq.randnums, self.seq.ylabels)), self.seq.label_groups)

    def test_shapes(self):
        self.assertEqual(
            self.seq.shape,
            ([(400, 36), (100, ), (100, ), (32, 32, 12), (34, 34, 11)], 4)
        )

    def test_num_batches(self):
        self.assertEqual(len(self.seq), 1)

    def test_get_index(self):
        xdata, ydata = self.seq[0]  # pylint: disable=unbalanced-tuple-unpacking
        self.assertEqual(xdata[0].shape, (10, 400, 36))
        self.assertEqual(xdata[1].shape, (10, 100, ))
        self.assertEqual(xdata[2].shape, (10, 100, ))
        self.assertEqual(xdata[3].shape, (10, 32, 32, 12, ))
        self.assertEqual(xdata[4].shape, (10, 34, 34, 11, ))
        self.assertEqual(ydata.shape, (10, 4))
