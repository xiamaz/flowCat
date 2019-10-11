"""Test single case generation and usage."""
import unittest
import datetime

from flowcat.dataset import case


def create_date(dstr):
    return datetime.datetime.strptime(dstr, "%Y-%m-%d").date()


class TestCaseGenerated(unittest.TestCase):
    """Test case creation from generated data."""

    def test_creation(self):
        """Simple creation using a dict of data."""
        tests = {
            "simple": {
                "date": "2018-10-11",
                "infiltration": 10.8,
                "cohort": "testcohort",
                "id": "12345",
                "diagnosis": "more stuff",
                "sureness": "suüre",
                "filepaths": [],
            },
            "minimum": {
                "id": "123145",
                "date": "2018-10-12",
                "filepaths": [],
            }
        }
        for tname, tdata in tests.items():
            with self.subTest(name=tname):
                tcase = case.caseinfo_to_case(tdata, "")
                self.assertEqual(tcase.id, tdata["id"])
                self.assertEqual(tcase.date, create_date(tdata["date"]))
                self.assertEqual(tcase.infiltration, tdata["infiltration"] if "infiltration" in tdata else 0.0)
                self.assertEqual(tcase.group, tdata["cohort"] if "cohort" in tdata else "")

        missing_tests = {
            "missingid": {
                "date": "2018-10-11",
                "infiltration": 10.8,
                "cohort": "testcohort",
                "diagnosis": "more stuff",
                "sureness": "suüre",
                "filepaths": [],
            },
            "missingdate": {
                "infiltration": 10.8,
                "cohort": "testcohort",
                "id": "12345",
                "diagnosis": "more stuff",
                "sureness": "suüre",
                "filepaths": [],
            },
            "missingfp": {
                "date": "2018-10-11",
                "infiltration": 10.8,
                "cohort": "testcohort",
                "id": "12345",
                "diagnosis": "more stuff",
                "sureness": "suüre",
            }
        }
        for tname, tdata in missing_tests.items():
            with self.subTest(tname):
                with self.assertRaisesRegex(AssertionError, r"^\w+ is required$"):
                    case.caseinfo_to_case(tdata, "")

        conversion_tests = {
            "strinfil": ({
                "date": "2018-10-11",
                "infiltration": "10.8",
                "cohort": "testcohort",
                "id": "12345",
                "diagnosis": "more stuff",
                "sureness": "suüre",
                "filepaths": [],
            }, 10.8),
            "commainfil": ({
                "date": "2018-10-11",
                "infiltration": "10,8",
                "cohort": "testcohort",
                "id": "12345",
                "diagnosis": "more stuff",
                "sureness": "suüre",
                "filepaths": [],
            }, 10.8),
            "intinfil": ({
                "date": "2018-10-11",
                "infiltration": 10,
                "cohort": "testcohort",
                "id": "12345",
                "diagnosis": "more stuff",
                "sureness": "suüre",
                "filepaths": [],
            }, 10.0),
            "outofrange": ({
                "date": "2018-10-11",
                "infiltration": 1000,
                "cohort": "testcohort",
                "id": "12345",
                "diagnosis": "more stuff",
                "sureness": "suüre",
                "filepaths": [],
            }, AssertionError),
            "outofrange2": ({
                "date": "2018-10-11",
                "infiltration": -1,
                "cohort": "testcohort",
                "id": "12345",
                "diagnosis": "more stuff",
                "sureness": "suüre",
                "filepaths": [],
            }, AssertionError),
        }
        for tname, (tdata, truth) in conversion_tests.items():
            with self.subTest(tname):
                if isinstance(truth, float):
                    tcase = case.caseinfo_to_case(tdata, "")
                    self.assertEqual(tcase.infiltration, truth)
                else:
                    with self.assertRaises(truth):
                        case.caseinfo_to_case(tdata, "")
                        case.Case(tdata)

    def test_samples(self):
        """Check correct creation of samples."""
        base_obj = {
            "id": "12345",
            "date": "2018-10-10",
        }

        tests = {
            "simple": [
                {
                    "date": "2018-10-12",
                    "fcs": {
                        "path": "nop",
                    },
                }
            ]
        }

        for tname, tpathdata in tests.items():
            with self.subTest(tname):
                tdata = {**base_obj, "filepaths": tpathdata}
                tcase = case.caseinfo_to_case(tdata, "")
                self.assertEqual(str(tcase.samples[0].path), tpathdata[0]["fcs"]["path"])
