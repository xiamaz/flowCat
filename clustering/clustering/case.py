import os
import logging
from enum import Enum

import fcsparser

from .utils import get_file_path


LOGGER = logging.getLogger(__name__)


def create_fcs_path(path: str):
    """Add s3 prefix to fcs paths."""
    return os.path.join("s3://mll-flowdata", path)


def all_in(smaller, larger):
    """Check that all items in the smaller iterable is in the larger iterable.
    """
    for item in smaller:
        if item not in larger:
            return False
    return True


class Material(Enum):
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


class Case:
    """Basic case object containing all metadata for a case."""
    def __init__(self, data: dict):

        self._filepaths = None
        self._tubepaths = None
        self._tube_markers = None

        # place to store result
        self._histogram = {}

        self.infiltration = data["infiltration"]
        self.group = data["cohort"]
        self.id = data["id"]

        self.filepaths = data["destpaths"]

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

    def get_tube(self, tube: int) -> dict:
        """Get filedict for a single tube. Return the last filedict in the
        list."""
        assert self.has_tube(tube), "Case does not have specified tube."
        all_tube = self.tubepaths[tube]
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


class CasePath:
    """Single path for a case."""
    def __init__(self, path, parent):
        self.path = create_fcs_path(path["path"])
        self.tube = int(path["tube"])
        self.markers = path["markers"]
        self.material = Material.from_str(path["material"])

        self.parent = parent

        self.result = None
        self._data = None

    @property
    def data(self):
        """FCS data."""
        if self._data is None:
            _, self._data = fcsparser.parse(
                get_file_path(self.path), data_set=0, encoding="latin-1"
            )
        return self._data

    @property
    def dict(self) -> dict:
        """Dict representation."""
        if self.result is None:
            raise RuntimeError("Result not generated for case path.")
        return {
            **dict(zip(range(len(self.result)), self.result)),
            **{
                "label": self.parent.id,
                "group": self.parent.group,
                "infiltration": self.parent.infiltration,
            }
        }

    def has_markers(self, markers: list) -> bool:
        """Return whether given list of markers are fulfilled."""
        return all_in(markers, self.markers)
