"""
Objects abstracting basic case information.

Look at tests/test_case.py to see how cases can be defined from simple dicts.
"""
from __future__ import annotations

import logging
import functools
import datetime
from dataclasses import dataclass, replace, field, asdict
from typing import List, Dict, Union, Tuple

from dataslots import with_slots
import pandas as pd

from flowcat import mappings, utils
from . import sample


LOGGER = logging.getLogger(__name__)


CASE_REQUIRED_FIELDS = "date", "id", "filepaths"


def assert_in_dict(fields, data):
    for sfield in fields:
        assert sfield in data, f"{field} is required"


def filter_case(
        case: Case,
        tubes: List[str] = None,
        labels: List[str] = None,
        groups: List[str] = None,
        infiltration: Tuple[float, float] = None,
        date: Tuple[Union[str], Union[str]] = None,
        counts: int = None,
        materials: List[str] = None,
        selected_markers: Dict[int, List[str]] = None,
) -> Tuple[bool, str]:
    """
    Args:
        case: Case object.
        tubes: List of tubes, such as [1, 2, 3].
        labels: List of case ids.
        groups: List of groups, such as [CLL, MBL, HCL].
        infiltration: Two numbers for lower and upper bound, eg (0.2, 0.8) or (0.2, None)
        date: Lower and upper date in YYYY-MM-DD format.
        counts: Minimum number of events in fcs file.
        materials: List of allowed materials.
        selected_markers: Dictionary mapping tubes to allowed markers.

    Returns:
        Tuple of whether case fulfills requirements and optionally the reason for exclusion.
    """
    reasons = []
    if groups and case.group not in groups:
        reasons.append(f"groups")

    if tubes and not case.has_tubes(tubes):
        reasons.append(f"tubes")

    if labels and case.id not in labels:
        reasons.append(f"labels")

    if infiltration and case.group != "normal":
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


def caseinfo_to_case(caseinfo: dict, sample_path: utils.URLPath) -> Case:
    # required keys
    assert_in_dict(CASE_REQUIRED_FIELDS, caseinfo)
    case_id = caseinfo["id"]
    case_date = utils.str_to_date(caseinfo["date"])
    samples = [
        sample.sampleinfo_to_sample(sampleinfo, case_id, sample_path)
        for sampleinfo in caseinfo["filepaths"]
    ]

    # optional keys
    infiltration = caseinfo.get("infiltration", 0.0)
    infiltration = float(infiltration.replace(",", ".") if isinstance(infiltration, str) else infiltration)
    assert infiltration <= 100.0 and infiltration >= 0.0, "Infiltration out of range 0-100"
    group = caseinfo.get("cohort", "")
    diagnosis = caseinfo.get("diagnosis", "")
    sureness = caseinfo.get("sureness", "")

    case = Case(
        id=case_id,
        date=case_date,
        samples=samples,
        infiltration=infiltration,
        group=group,
        diagnosis=diagnosis,
        sureness=sureness,
    )
    return case


def case_to_json(case: Case) -> dict:
    casedict = asdict(case)
    if casedict["used_material"]:
        casedict["used_material"] = casedict["used_material"].name
    else:
        casedict["used_material"] = ""
    casedict["date"] = case.date.isoformat()
    casedict["samples"] = [sample.sample_to_json(s) for s in case.samples]
    return casedict


def json_to_case(jscase: dict) -> Case:
    if jscase["used_material"]:
        material = mappings.Material[jscase["used_material"]]
    else:
        material = None
    jscase["used_material"] = material
    jscase["date"] = utils.str_to_date(jscase["date"])

    return Case(**jscase)


@with_slots
@dataclass
class Case:
    """Basic case object containing all metadata for a case."""
    id: str
    used_material: mappings.Material = None
    date: datetime.date = None
    infiltration: float = None
    diagnosis: str = None
    sureness: str = None
    group: str = None
    samples: List[sample.Sample] = field(default_factory=list)

    def set_fcs_info(self):
        """Load fcs information for all available samples from fcs files."""
        for case_sample in self.samples:
            case_sample.set_fcs_info()

    def get_markers(self, tubes):
        """Return a dictionary of markers."""
        return {tube: self.get_tube_markers(tube) for tube in tubes}

    def get_tube_markers(self, tube):
        tube = self.get_tube(tube)
        if tube is None:
            return []
        return tube.markers

    def get_tube_samples(self, tube: str, kind: str) -> List[sample.Sample]:
        if kind == "fcs":
            samples = [s for s in self.samples if isinstance(s, sample.FCSSample) and s.tube == tube]
        elif kind == "som":
            samples = [s for s in self.samples if isinstance(s, sample.SOMSample) and s.tube == tube]
        else:
            raise ValueError(f"{kind} is not known. Valid options are: fcs, som")
        return samples

    def get_tube(self, tube: str, kind: str = "fcs") -> sample.Sample:
        """Get the TubePath fulfilling the given requirements, return the
        last on the list if multiple are available.
        Args:
            tube: Tube number to be selected.
            kind: Kind of sample to return.
        Returns:
            TubePath or None.
        """
        samples = self.get_tube_samples(tube, kind)
        if len(samples) != 1:
            raise RuntimeError(f"Ambiguous {self.id}. Needed one, but got {len(samples)}")
        return samples[0]

    def get_same_materials(self, tubes):
        """Check whether the given tubes can have the same material.
        Args:
            tubes: List of ints of needed tubes.
        Returns:
            List of possible materials.
        """
        materials = [
            set(f.material for f in self.samples
                if f.tube == t) for t in tubes
        ]

        found_all = functools.reduce(lambda x, y: x & y, materials)
        return found_all

    def has_markers(self, tube, markers):
        try:
            tcase = self.get_tube(tube)
        except RuntimeError:
            tcase = None
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
        available_tubes = set(f.tube for f in self.samples)
        # check if all wanted tubes are inside available tubes using set ops
        return wanted_tubes <= available_tubes

    def get_merged_data(self, tubes, channels=None, **kwargs):
        """Get dataframe from selected tubes and channels.
        """
        sel_tubes = [self.get_tube(t).get_data(**kwargs) for t in tubes]
        joined = pd.concat(sel_tubes, sort=False)
        if channels:
            joined = joined[channels]
        else:
            joined = joined[[c for c in joined.columns if "nix" not in c]]
        return joined

    def copy(self, **replaced):
        """Copy the object and replace the given arguments."""
        return replace(self, **replaced)

    def __repr__(self):
        return f"<Case {self.id}| G{self.group} {self.date} {len(self.samples)} files>"
