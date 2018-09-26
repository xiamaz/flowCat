import os
import logging
import enum
from datetime import datetime

from sklearn import preprocessing
import pandas as pd

import fcsparser

from .utils import get_file_path


LOGGER = logging.getLogger(__name__)


def all_in(smaller, larger):
    """Check that all items in the smaller iterable is in the larger iterable.
    """
    for item in smaller:
        if item not in larger:
            return False
    return True


class Sureness(enum.IntEnum):
    HIGH = 10
    NORMAL = 5
    LOW = 1


class Material(enum.Enum):
    """Class containing material types. Abstracting the concept for
    easier consumption."""
    PERIPHERAL_BLOOD = 1
    BONE_MARROW = 2
    OTHER = 3

    @staticmethod
    def from_str(label: str) -> "Material":
        if label in ["1", "2", "3", "4", "5", "PB"]:
            return Material.PERIPHERAL_BLOOD
        elif label == "KM":
            return Material.BONE_MARROW
        else:
            return Material.OTHER


class Case(object):
    """Basic case object containing all metadata for a case."""
    __slots__ = (
        "_json",
        "path",
        "_filepaths",
        "_tubepaths",
        "_tube_markers",
        "_histogram",
        "date",
        "infiltration",
        "short_diagnosis",
        "sureness_description",
        "sureness",
        "group",
        "id",
    )

    def __init__(self, data: dict, path: str = ""):
        self._json = data

        self.path = path

        self._filepaths = None
        self._tubepaths = None
        self._tube_markers = None

        # place to store result
        self._histogram = {}

        self.date = datetime.strptime(data["date"], "%Y-%m-%d").date()

        self.infiltration = data["infiltration"]
        self.group = data["cohort"]
        self.id = data["id"]

        self.filepaths = data["filepaths"]

        self.short_diagnosis = data["diagnosis"]
        self.sureness_description = data["sureness"]
        self.sureness = self._infer_sureness()

    @property
    def json(self):
        return self._json

    @property
    def filepaths(self):
        """Get a list of filepaths."""
        return self._filepaths

    @filepaths.setter
    def filepaths(self, value: list):
        """Set filepaths and clear all generated dicts on data."""
        self._filepaths = [
            CasePath(v, self) if not isinstance(v, CasePath) else v
            for v in value
        ]
        self._tubepaths = None
        self._tube_markers = None

    @property
    def tubepaths(self) -> dict:
        """Dict of tubepath ids to list of filedicts."""
        if self._tubepaths is None:
            self._tubepaths = {
                t: [fp for fp in self.filepaths if t == int(fp.tube)]
                for t in set([int(fp.tube) for fp in self.filepaths])
            }
        return self._tubepaths

    @property
    def tube_markers(self) -> dict:
        """Dict of tube to selected marker lists."""
        if self._tube_markers is None:
            self._tube_markers = {
                k: v.markers
                for k, v in
                {k: self.get_tube(k) for k in self.tubepaths}.items()
                if v
            }
        return self._tube_markers

    def get_tube(self, tube: int, min_count: int = 0) -> dict:
        """Get filedict for a single tube. Return the last filedict in the
        list."""
        assert self.has_tube(tube), "Case does not have specified tube."
        all_tube = self.tubepaths[tube]
        if min_count:
            for tcase in all_tube:
                if tcase.event_count > min_count:
                    return tcase
            raise RuntimeError(
                f"No case found fulfilling {min_count} in {tube}")
        return all_tube[-1]

    def has_tube(self, tube: int) -> bool:
        """Check whether case has a specified tube.
        """
        return bool(self.tubepaths.get(tube, []))

    def get_tube_markers(self, tube: int) -> list:
        """Get markers for the given tube."""
        return self.tube_markers.get(tube, [])

    def has_tubes(self, tubes: list):
        """Check that a Case has all given tubes.
        """
        return all([self.has_tube(t) for t in tubes])

    def same_material(self, tubes: list):
        """Check that the materials returned for the
        list of given tubes are of the same material"""
        material_num = len(
            {self.get_tube(t).material for t in tubes}
        )
        return material_num == 1

    def has_count(self, count: int, tubes: list):
        """Check if case has the required counts in the needed channels."""
        return all(any(p.event_count >= count for p in v) for v in self.tubepaths.values())

    def get_merged_data(self, tubes=None, channels=None, min_count=0, **kwargs):
        """Get dataframe from selected tubes and channels.
        """
        tubes = sorted(list(self.tube_markers.keys())) if tubes is None else tubes
        sel_tubes = [self.get_tube(t, min_count=min_count).get_data(**kwargs) for t in tubes]
        joined = pd.concat(sel_tubes, sort=False)
        if channels:
            joined = joined[channels]
        else:
            joined = joined[[c for c in joined.columns if "nix" not in c]]
        return joined

    def _infer_sureness(self):
        """Return a sureness score from existing information."""
        sureness_desc = self.sureness_description.lower()
        short_diag = self.short_diagnosis.lower()
        if self.group == "FL":
            if "nachweis eines igh-bcl2" in sureness_desc:
                return Sureness.HIGH
            else:
                return Sureness.NORMAL
        elif self.group == "MCL":
            if "mit nachweis ccnd1-igh" in sureness_desc:
                return Sureness.HIGH
            elif "nachweis eines igh-ccnd1" in sureness_desc:  # synon. to first
                return Sureness.HIGH
            elif "nachweis einer 11;14-translokation" in sureness_desc:  # synon. to first
                return Sureness.HIGH
            elif "mantelzelllymphom" in short_diag:  # prior known diagnosis will be used
                return Sureness.HIGH
            elif "ohne fish-sonde" in sureness_desc:  # diagnosis uncertain without genetic proof
                return Sureness.LOW
            else:
                return Sureness.NORMAL
        elif self.group == "PL":
            if "kein nachweis eines igh-ccnd1" in sureness_desc:  # hallmark MCL (synon. 11;14)
                return Sureness.HIGH
            elif "kein nachweis einer 11;14-translokation" in sureness_desc:  # synon to first
                return Sureness.HIGH
            elif "nachweis einer 11;14-translokation" in sureness_desc:  # hallmark MCL
                return Sureness.LOW
            else:
                return Sureness.NORMAL
        elif self.group == "LPL":
            if "lymphoplasmozytisches lymphom" in short_diag:  # prior known diagnosis will be used
                return Sureness.HIGH
            else:
                return Sureness.NORMAL
        elif self.group == "MZL":
            if "marginalzonenlymphom" in short_diag:  # prior known diagnosis will be used
                return Sureness.HIGH
            else:
                return Sureness.NORMAL
        else:
            return Sureness.NORMAL


class CasePath(object):
    __slots__ = (
        "path",
        "markers",
        "event_count",
        "tube",
        "material",
        "parent",
        "result",
        "result_success",
    )

    """Single path for a case."""
    def __init__(self, path, parent):
        self.path = os.path.join(parent.path, path["fcs"]["path"])
        self.markers = path["fcs"]["markers"]
        self.event_count = path["fcs"]["event_count"]

        self.tube = int(path["tube"])
        self.material = Material.from_str(path["material"])

        self.parent = parent

        self.result = None
        self.result_success = False

    @property
    def data(self):
        """FCS data. Do not save the fcs data in the case, since
        it would be too large."""
        return self.get_data(normalized=False, scaled=False)

    def get_data(self, normalized=True, scaled=True):
        """
        Args:
            normalized: Normalize data to mean and standard deviation.
            scaled: Scale data between 0 and 1.
        Returns:
            Dataframe with fcs data.
        """
        _, data = fcsparser.parse(
            get_file_path(self.path), data_set=0, encoding="latin-1")

        if normalized:
            data = pd.DataFrame(
                preprocessing.StandardScaler().fit_transform(data), columns=data.columns)
        if scaled:
            data = pd.DataFrame(
                preprocessing.MinMaxScaler().fit_transform(data), columns=data.columns)

        return data

    @property
    def metainfo_dict(self):
        """Return case metainformation."""
        return {
            "label": self.parent.id,
            "group": self.parent.group,
            "infiltration": self.parent.infiltration,
        }

    @property
    def dict(self) -> dict:
        """Dict representation."""
        if self.result is None:
            raise RuntimeError("Result not generated for case path.")
        return {
            **dict(zip(range(len(self.result)), self.result)),
            **self.metainfo_dict,
        }

    @property
    def fail_dict(self) -> dict:
        """Dict representation of failure messages."""
        return {
            **{
                "status": self.result_success,
                "reason": self.result if isinstance(self.result, str) else "",
            },
            **self.metainfo_dict,
        }

    def has_markers(self, markers: list) -> bool:
        """Return whether given list of markers are fulfilled."""
        return all_in(markers, self.markers)
