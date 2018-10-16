import unittest
import pathlib

from flowcat import utils
from flowcat.data import all_dataset
from flowcat.mappings import GROUP_MAPS


TESTPATH = pathlib.Path(__file__).parent
DATAPATH = TESTPATH / "data"
TMPPATH = TESTPATH / "tmp"

utils.TMP_PATH = str(TMPPATH)

FCS_PATH = DATAPATH / "small_dataset"
SOM_PATH = DATAPATH / "som" / "testrun_s32_ttoroid"
HISTO_PATH = DATAPATH / "histogram" / "abstract_normal"


class TestCombinedDataset(unittest.TestCase):

    def setUp(self):
        self.data = all_dataset.CombinedDataset.from_paths(
            casepath=FCS_PATH, paths=[("HISTO", HISTO_PATH), ("SOM", SOM_PATH)]
        )

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
