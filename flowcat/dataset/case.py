"""
Objects abstracting basic case information.

Look at tests/test_case.py to see how cases can be defined from simple dicts.
"""
from __future__ import annotations

import logging
import functools
from typing import List, Dict, Union, Tuple
from datetime import datetime

import pandas as pd

from .. import mappings, utils
from . import fcs


LOGGER = logging.getLogger(__name__)


CASE_REQUIRED_FIELDS = "date", "id", "filepaths"


def all_in(smaller, larger):
    """Check that all items in the smaller iterable is in the larger iterable.
    """
    for item in smaller:
        if item not in larger:
            return False
    return True


def assert_in_dict(fields, data):
    for field in fields:
        assert field in data, f"{field} is required"


def filter_case(
        case: Case,
        tubes: List[int] = None,
        labels: List[str] = None,
        groups: List[str] = None,
        infiltration: Tuple[float, float] = None,
        date: Tuple[Union[str], Union[str]] = None,
        counts: int = None,
        materials: List[str] = None,
        selected_markers: Dict[int, List[str]] = None,
) -> Tuple[bool, str]:
    reasons = []
    if groups and case.group not in groups:
        reasons.append(f"groups")

    if tubes and not case.has_tubes(tubes):
        reasons.append(f"tubes")

    if labels and case.id not in labels:
        reasons.append(f"labels")

    if infiltration:
        infiltration_min, infiltration_max = infiltration
        if infiltration_min and case.infiltration < infiltration_min:
            reasons.append("infiltration_min")
        if infiltration_max and case.infiltration > infiltration_max:
            reasons.append("infiltration_max")

    if date:
        date_min, date_max = date
        if date_min:
            date_min = utils.str_to_date(date_min) if isinstance(date_min, str) else date
            if case.date < date_min:
                reasons.append("date_min")
        if date_max:
            date_max = utils.str_to_date(date_max) if isinstance(date_max, str) else date
            if case.date > date_max:
                reasons.append("date_max")

    if counts:
        assert tubes, "Tubes required for counts"
        if not any(case.get_tube(t, min_count=counts) for t in tubes):
            reasons.append("counts")

    if materials:
        assert tubes, "Tubes required for materials"
        if not case.has_same_material(tubes, allowed_materials=materials):
            reasons.append("materials")

    if selected_markers and not case.has_selected_markers(selected_markers):
        reasons.append("selected_markers")

    return not bool(reasons), reasons


class Case:
    """Basic case object containing all metadata for a case."""
    __slots__ = (
        "_json",
        "path",
        "_filepaths",
        "used_material",
        "date",
        "infiltration",
        "sureness",
        "group",
        "id",
    )

    def __init__(self, data, path=""):
        """
        Args:
            data: Contains all metainformation, either a dictionary or
                a case object.
            path: Path prefix used for loading any data.
        Returns:
            New case object.
        """
        self._filepaths = None
        self.used_material = None

        if isinstance(data, self.__class__):
            self._json = data.raw.copy()
            self.path = path if path else data.path

            self.id = data.id
            self.date = data.date
            self.filepaths = data.filepaths

            self.infiltration = data.infiltration
            self.group = data.group
            self.sureness = data.sureness
        elif isinstance(data, dict):
            self._json = data
            self.path = path

            # required keys
            assert_in_dict(CASE_REQUIRED_FIELDS, data)
            self.id = data["id"]
            self.date = utils.str_to_date(data["date"])
            self.filepaths = data["filepaths"]

            # optional keys
            infiltration = data.get("infiltration", 0.0)
            self.infiltration = float(
                infiltration.replace(",", ".") if isinstance(infiltration, str) else infiltration)
            assert self.infiltration <= 100.0 and self.infiltration >= 0.0, "Infiltration out of range 0-100"
            self.group = data.get("cohort", "")
            short_diagnosis = data.get("diagnosis", "")
            sureness = data.get("sureness", "")
            self.sureness = self._infer_sureness(sureness, short_diagnosis)
        else:
            raise TypeError("data needs to be either another Case or a dict")

    @property
    def json(self):
        """Get dict representation of data usable for saving to json."""
        cdict = {
            "id": self.id,
            "date": self.date.isoformat(),
            "filepaths": [p.json for p in self.filepaths],
            "cohort": self.group,
            "infiltration": self.infiltration,
            "diagnosis": self.raw["diagnosis"],
            "sureness": self.raw["sureness"],
        }
        return cdict

    @property
    def raw(self):
        """Return raw input dictionary."""
        return self._json

    @property
    def filepaths(self):
        """Get a list of filepaths."""
        return self._filepaths

    @filepaths.setter
    def filepaths(self, value: list):
        """Set filepaths and clear all generated dicts on data.
        Args:
            value: List of dictionaries with information to create case paths
            or case paths.
        """
        self._filepaths = [TubeSample(v, self) for v in value]

    def set_markers(self):
        for fp in self.filepaths:
            fp.load()
            fp.set_markers()

    def get_markers(self, tubes):
        """Return a dictionary of markers."""
        return {tube: self.get_tube_markers(tube) for tube in tubes}

    def get_tube_markers(self, tube):
        tube = self.get_tube(tube)
        if tube is None:
            return [];
        return tube.markers

    def get_tube(self, tube, min_count=0, materials=None):
        """Get the TubePath fulfilling the given requirements, return the
        last on the list if multiple are available.
        Args:
            tube: Int tube number to be selected.
            min_count: Minimum number of events in the FCS file.
            material: Type of material used for the tube.
        Returns:
            TubePath or None.
        """
        if self.used_material and not materials:
            materials = [self.used_material]

        tubecases = [
            p for p in self.filepaths
            if p.tube == tube and (
                (not materials) or (p.material in materials)
            ) and (
                not min_count or min_count <= p.count)
        ]
        return tubecases[-1] if tubecases else None

    def get_same_materials(self, tubes):
        """Check whether the given tubes can have the same material.
        Args:
            tubes: List of ints of needed tubes.
        Returns:
            List of possible materials.
        """
        materials = [
            set(f.material for f in self.filepaths
                if f.tube == t) for t in tubes
        ]

        found_all = functools.reduce(lambda x, y: x & y, materials)
        return found_all

    def has_markers(self, tube, markers):
        tcase = self.get_tube(tube)
        if tcase is None:
            return False;
        return tcase.has_markers(markers)

    def has_selected_markers(self, selected_markers):
        """Check whether default samples can fulfill the selected markers."""
        return all(self.has_markers(t, m) for t, m in selected_markers.items())

    def get_possible_material(self, tubes, allowed_materials=None):
        """Get one possible material for the given case.
        """
        available_materials = self.get_same_materials(tubes)
        if allowed_materials:
            filtered = [m for m in available_materials if m in allowed_materials]
        else:
            filtered = available_materials
        return filtered[0] if filtered else None

    def set_allowed_material(self, tubes):
        """Set used material to one of the allowed materials."""
        self.used_material = self.get_possible_material(tubes, mappings.ALLOWED_MATERIALS)

    def has_same_material(self, tubes, allowed_materials=None):
        if allowed_materials is None:
            return bool(self.get_same_materials(tubes))
        return bool(self.get_possible_material(tubes, allowed_materials))

    def has_tubes(self, tubes):
        """Check whether case has the specified tube or tubes.
        Args:
            tubes: Int or list of ints for the required tubes.
            same_material: Check whether the possible subselection has the same
                material.
        Returns:
            True if the required tubes are available.
        """
        wanted_tubes = set(tubes)
        available_tubes = set(f.tube for f in self.filepaths)
        # check if all wanted tubes are inside available tubes using set ops
        return wanted_tubes <= available_tubes

    def get_merged_data(self, tubes, channels=None, min_count=0, **kwargs):
        """Get dataframe from selected tubes and channels.
        """
        sel_tubes = [self.get_tube(t, min_count=min_count).get_data(**kwargs) for t in tubes]
        joined = pd.concat(sel_tubes, sort=False)
        if channels:
            joined = joined[channels]
        else:
            joined = joined[[c for c in joined.columns if "nix" not in c]]
        return joined

    def _infer_sureness(self, sureness_desc, short_diag):
        """Return a sureness score from existing information."""
        sureness_desc = sureness_desc.lower()
        short_diag = short_diag.lower()

        if self.group == "FL":
            if "nachweis eines igh-bcl2" in sureness_desc:
                return mappings.Sureness.HIGH
            return mappings.Sureness.NORMAL
        if self.group == "MCL":
            if "mit nachweis ccnd1-igh" in sureness_desc:
                return mappings.Sureness.HIGH
            if "nachweis eines igh-ccnd1" in sureness_desc:  # synon. to first
                return mappings.Sureness.HIGH
            if "nachweis einer 11;14-translokation" in sureness_desc:  # synon. to first
                return mappings.Sureness.HIGH
            if "mantelzelllymphom" in short_diag:  # prior known diagnosis will be used
                return mappings.Sureness.HIGH
            if "ohne fish-sonde" in sureness_desc:  # diagnosis uncertain without genetic proof
                return mappings.Sureness.LOW
            return mappings.Sureness.NORMAL
        if self.group == "PL":
            if "kein nachweis eines igh-ccnd1" in sureness_desc:  # hallmark MCL (synon. 11;14)
                return mappings.Sureness.HIGH
            if "kein nachweis einer 11;14-translokation" in sureness_desc:  # synon to first
                return mappings.Sureness.HIGH
            if "nachweis einer 11;14-translokation" in sureness_desc:  # hallmark MCL
                return mappings.Sureness.LOW
            return mappings.Sureness.NORMAL
        if self.group == "LPL":
            if "lymphoplasmozytisches lymphom" in short_diag:  # prior known diagnosis will be used
                return mappings.Sureness.HIGH
            return mappings.Sureness.NORMAL
        if self.group == "MZL":
            if "marginalzonenlymphom" in short_diag:  # prior known diagnosis will be used
                return mappings.Sureness.HIGH
            return mappings.Sureness.NORMAL
        return mappings.Sureness.NORMAL

    def copy(self):
        return self.__class__(self)

    def __repr__(self):
        return f"<Case {self.id}| G{self.group} {self.date} {len(self.filepaths)} files>"


class TubeSample:
    """FCS sample metadata wrapper."""
    __slots__ = (
        "_json",
        "_data",
        "count",
        "date",
        "path",
        "markers",
        "material",
        "parent",
        "panel",
        "tube",
    )

    """Single sample from a certain tube."""
    def __init__(self, data, parent):
        self._data = None
        if isinstance(data, self.__class__):
            self._json = data.raw
            self.path = data.path

            self.tube = data.tube
            self.material = data.material
            self.panel = data.panel
            self.markers = data.markers.copy() if data.markers is not None else None
            self.count = data.count
            self.date = data.date
        else:
            self._json = data
            assert "fcs" in data and "path" in data["fcs"], "Path to data is missing"
            assert "date" in data, "Date is missing"
            self.path = data["fcs"]["path"]
            self.date = datetime.strptime(data["date"], "%Y-%m-%d").date()

            self.tube = int(data.get("tube", 0))
            self.material = mappings.Material.from_str(data.get("material", ""))
            self.panel = data.get("panel", "")

            self.markers = data["fcs"].get("markers", None)
            self.count = int(data["fcs"].get("event_count", 0))

        self.parent = parent

    @property
    def raw(self):
        return self._json

    @property
    def json(self):
        return {
            "id": self.parent.id,
            "panel": self.panel,
            "tube": self.tube,
            "date": self.date.isoformat(),
            "material": self.raw["material"],
            "fcs": {
                "path": self.path,
                "markers": self.markers,
                "event_count": self.count,
            }
        }

    @property
    def data(self):
        """FCS data. Do not save the fcs data in the case, since
        it would be too large."""
        if self._data is not None:
            return self._data
        return self.get_data(normalized=False, scaled=False)

    @property
    def urlpath(self):
        url_path = utils.URLPath(self.parent.path) / self.path
        return url_path

    @property
    def localpath(self):
        local_path = self.urlpath.get()
        return local_path

    def get_data(self, normalized=True, scaled=True):
        """
        Args:
            normalized: Normalize data to mean and standard deviation.
            scaled: Scale data between 0 and 1.
        Returns:
            Dataframe with fcs data.
        """
        url_path = utils.URLPath(self.parent.path) / self.path
        data = fcs.FCSData(url_path.get())
        if normalized:
            data = data.normalize()
        if scaled:
            data = data.scale()
        return data

    def load(self):
        """Load data into slot."""
        self._data = self.get_data()
        return self._data

    def clear(self):
        """Clear data from slot."""
        self._data = None

    def set_markers(self):
        """Set markers from own data."""
        assert self._data is not None, "Load data beforehand into slot"
        self.markers = list(self.data.channels)

    def has_markers(self, markers: list) -> bool:
        """Return whether given list of markers are fulfilled."""
        return all_in(markers, self.markers)

    def __repr__(self):
        return f"<Sample| T{self.tube} D{self.date} {self.count} events>"
