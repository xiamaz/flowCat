"""
Test creation of SOM from initial raw data.
"""
import unittest

from pandas.testing import assert_frame_equal

from flowcat import som, configuration, utils
from flowcat.dataset import case_dataset, case

from . import shared


CHANNELS = {
    1: [
        "FS INT LIN",
        "SS INT LIN",
        "FMC7-FITC",
        "CD10-PE",
        "IgM-ECD",
        "CD79b-PC5.5",
        "CD20-PC7",
        "CD23-APC",
        "CD19-APCA750",
        "CD5-PacBlue",
        "CD45-KrOr",
    ],
    2: [
        "FS INT LIN",
        "SS INT LIN",
        "Kappa-FITC",
        "Lambda-PE",
        "CD38-ECD",
        "CD25-PC5.5",
        "CD11c-PC7",
        "CD103-APC",
        "CD19-APCA750",
        "CD22-PacBlue",
        "CD45-KrOr",
    ],
}


def create_single_case(name):
    filepaths = [
        {"fcs": {"path": f"{name}_{i}.lmd"}, "date": "2018-01-02", "tube": i} for i in [1, 2]
    ]
    tdict = {
        "id": name,
        "date": "2018-01-02",
        "filepaths": filepaths,
    }
    return case.Case(tdict, path=shared.DATAPATH / "fcs")


def create_configuration():

    return configuration.SOMConfig({
        "name": "testconfig",
        "dataset": {
            "names": {},
            "filters": {
                "tubes": [1, 2],
            },
            "selected_markers": CHANNELS,
        },
        "tfsom": {
            "model_name": "testmodel",
        }
    })


def create_configuration_reference(refpath):
    return configuration.SOMConfig({
        "name": "testconfig",
        "dataset": {
            "names": {},
            "filters": {
                "tubes": [1, 2],
            },
            "selected_markers": CHANNELS,
        },
        "reference": str(refpath),
        "tfsom": {
            "model_name": "testmodel",
            "initialization_method": "reference",
        }
    })


def assert_som_equal(som_a, som_b, tubes=None):
    if tubes is None:
        tubes = [1, 2]
    for tube in tubes:
        assert_frame_equal(som_a[tube], som_b[tube], check_dtype=False)


class TestSOMCreation(unittest.TestCase):
    """Multiple SOM creation scenarios.

    Check for reproducibility with a predefined seed will be checked with
    different runs on the same machine.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = case_dataset.CaseCollection.from_path(
            shared.get_test_dataset("small_dataset"))

    def test_single_case_som(self):
        """Create a SOM using a single case. Check that results using the
        same seed stay the same."""
        tcase = create_single_case("cll1")
        config = create_configuration()

        for seed in [42, 32]:
            with self.subTest(seed=seed):
                result_a = som.create_som([tcase], config, seed=seed)
                result_b = som.create_som([tcase], config, seed=seed)
                assert_som_equal(result_a, result_b)

        with self.subTest("different seeds"):
            result_a = som.create_som([tcase], config, seed=20)
            result_b = som.create_som([tcase], config, seed=10)
            with self.assertRaises(AssertionError):
                assert_som_equal(result_a, result_b)

    def test_reference_initialization(self):
        """Generate SOM using weights from previous SOM for initialization."""
        tcase = create_single_case("cll2")
        config = create_configuration_reference(shared.DATAPATH / "som_seed42")
        reference = som.load_som(shared.DATAPATH / "som_seed42", [1, 2], suffix=False)

        with self.subTest("seeded"):
            result_a = som.create_som([tcase], config, seed=42, reference=reference)
            result_b = som.create_som([tcase], config, seed=42, reference=reference)
            assert_som_equal(result_a, result_b)

        # check that not using a reference will consistently return different
        # values
        with self.subTest("no reference"):
            nconfig = config.copy()
            nconfig.data["tfsom"]["initialization_method"] = "random"
            result_nonref = som.create_som([tcase], nconfig, seed=42, reference=None)
            with self.assertRaises(AssertionError):
                assert_som_equal(result_a, result_nonref)

        # check that training for 0 epochs will return the original reference
        with self.subTest("no training"):
            nconfig = config.copy()
            nconfig.data["tfsom"]["max_epochs"] = 0
            result_same = som.create_som([tcase], nconfig, seed=42, reference=reference)
            assert_som_equal(result_same, reference)

    def test_indiv_soms(self):
        """Generation of multiple SOMs from one reference."""
        sompath = utils.URLPath(shared.DATAPATH / "indivsom")

        config = create_configuration_reference(shared.DATAPATH / "som_seed42")
        reference = som.load_som(config("reference"), [1, 2], suffix=False)

        data = self.cases.filter(tubes=[1, 2])
        result = som.create_indiv_soms(data, config,  sompath, reference=reference)
        config.to_file(sompath / "config.toml")
        utils.save_csv(result, sompath + ".csv")
