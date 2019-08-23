"""
Sample contains metadata and produces the associated FCSData.
Since FCS files are quite large, these have to be explicitly loaded by
calling Sample.get_fcs().
"""
# pylint: skip-file
# flake8: noqa
from __future__ import annotations

from typing import List
from dataclasses import dataclass, replace

from dataslots import with_slots

from flowcat import mappings, utils
from flowcat.som import base
from . import fcs


def _all_in(smaller, larger):
    """Check that all items in the smaller iterable is in the larger iterable.
    """
    for item in smaller:
        if item not in larger:
            return False
    return True


def filter_samples(
        samples: List[Sample],
        tubes: List[int] = None,
        materials: List[str] = None,
        selected_markers: Dict[int, List[str]] = None,
        counts: int = None,
        **_
) -> List[Sample]:
    filtered = []
    for sample in samples:
        if tubes and sample.tube not in tubes:
            continue
        if materials and sample.material not in materials:
            continue
        if counts and sample.count < counts:
            continue
        tube_markers = selected_markers.get(sample.tube) if selected_markers else None
        if tube_markers and not sample.has_markers(tube_markers):
            continue
        filtered.append(sample)
    return filtered


def sampleinfo_to_sample(sample_info: dict, sample_id: str, path: utils.URLPath) -> Sample:
    """Create a tube sample from sample info dict."""
    assert "fcs" in sample_info and "path" in sample_info["fcs"], "Path to sample_info is missing"
    assert "date" in sample_info, "Date is missing"
    path = path / utils.URLPath(sample_info["fcs"]["path"])
    date = utils.str_to_date(sample_info["date"])

    tube = str(sample_info.get("tube", "0"))
    material = mappings.Material.from_str(sample_info.get("material", ""))
    panel = sample_info.get("panel", "")

    markers = sample_info["fcs"].get("markers", None)
    count = int(sample_info["fcs"].get("event_count", 0)) or None

    sample = FCSSample(
        id=sample_id,
        path=path,
        date=date,
        tube=tube,
        material=material,
        panel=panel,
        markers=markers,
        count=count)
    return sample


def sample_to_sampleinfo(sample: FCSSample, parent_path: utils.URLPath) -> dict:
    """Generate caseinfo from tubesample object."""
    return {
        "id": sample.id,
        "panel": sample.panel,
        "tube": sample.tube,
        "date": sample.date.isoformat(),
        "material": sample.material.name,
        "fcs": {
            "path": str(sample.path.relative_to(parent_path)),
            "markers": sample.markers,
            "event_count": sample.count,
        }
    }


@with_slots
@dataclass
class Sample:
    """Single sample from a certain tube."""
    id: str
    date: date
    tube: str
    path: utils.URLPath
    panel: str = None
    markers: List[str] = None
    material: mappings.Material = None
    count: int = None

    def has_markers(self, markers: list) -> bool:
        """Return whether given list of markers are fulfilled."""
        return _all_in(markers, self.markers)

    def __repr__(self):
        return f"<Sample {self.material}| T{self.tube} D{self.date} {self.count} events>"

    def copy(self, **replaced):
        return replace(self, **replaced)


@with_slots
@dataclass
class FCSSample(Sample):

    def get_data(self) -> fcs.FCSData:
        """
        Args:
            normalized: Normalize data to mean and standard deviation.
            scaled: Scale data between 0 and 1.
        Returns:
            Dataframe with fcs data.
        """
        data = fcs.FCSData(self.path)
        return data

    def set_fcs_info(self):
        data = self.get_data()
        self.markers = list(data.channels)
        self.count = data.shape[0]


@with_slots
@dataclass
class SOMSample(Sample):
    data: base.SOM = None

    def set_data(self, data: base.SOM):
        """Set the given data, also add metadata here to SOM."""
        if data.material is None:
            data.material = self.material
        else:
            if data.material != self.material:
                raise ValueError(f"{data} mismatch with {self}")
        if data.tube is None:
            data.tube = self.tube
        else:
            if data.tube != self.tube:
                raise ValueError(f"{data} mismatch with {self}")
        if not data.cases:
            data.cases = [self.id]
            if len(data.cases) != 1 or data.cases[0] != self.id:
                raise ValueError(f"{data} mismatch with {self}")
        self.data = data

    def get_data(self) -> base.SOM:
        """
        Args:
            normalized: Normalize data to mean and standard deviation.
            scaled: Scale data between 0 and 1.
        Returns:
            Dataframe with fcs data.
        """
        data = base.SOM.from_path(self.path)
        data.material = self.material
        data.tube = self.tube
        data.cases = [self.id]
        return data
