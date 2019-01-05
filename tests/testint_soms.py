"""
Test creation of SOM from initial raw data.
"""
import unittest

from pandas.testing import assert_frame_equal

from flowcat import som, configuration
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


class TestSOMCreation(unittest.TestCase):
    """Multiple SOM creation scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.cases = case_dataset.CaseCollection.from_path(
            shared.get_test_dataset("small_dataset"))

    def test_single_case_som(self):
        """Create a SOM using a single case."""
        orig = som.load_som(shared.DATAPATH / "som_seed42", [1, 2], suffix=False)

        tcase = create_single_case("cll1")
        config = create_configuration()
        result = som.create_som([tcase], config, seed=42)

        assert_frame_equal(orig[1], result[1], check_dtype=False)
        assert_frame_equal(orig[2], result[2], check_dtype=False)

    def test_reference_initialization(self):
        """Generate SOM using weights from previous SOM for initialization."""
        orig = som.load_som(shared.DATAPATH / "ref_som_seed42", [1, 2], suffix=False)

        tcase = create_single_case("cll2")
        config = create_configuration_reference(shared.DATAPATH / "som_seed42")
        result = som.create_som([tcase], config, seed=42)

        assert_frame_equal(orig[1], result[1], check_dtype=False)
        assert_frame_equal(orig[2], result[2], check_dtype=False)
