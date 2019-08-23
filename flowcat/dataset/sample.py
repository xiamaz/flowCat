"""
Sample contains metadata and produces the associated FCSData.
Since FCS files are quite large, these have to be explicitly loaded by
calling Sample.get_fcs().
"""
# pylint: skip-file
# flake8: noqa
from __future__ import annotations

from typing import List, Tuple, Union
from dataclasses import dataclass, replace, field, asdict

from dataslots import with_slots

from flowcat import mappings, utils
from . import fcs, som


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


def sampleinfo_to_sample(sample_info: dict, case_id: str, path: utils.URLPath) -> Sample:
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

    sample_id = f"{case_id}_t{tube}_{material.name}_{sample_info['date']}"

    sample = FCSSample(
        id=sample_id,
        case_id=case_id,
        path=path,
        date=date,
        tube=tube,
        material=material,
        panel=panel,
        markers=markers,
        count=count)
    return sample


def sample_to_json(sample: Sample) -> dict:
    if isinstance(sample, FCSSample):
        return {"__fcssample__": fcssample_to_json(sample)}
    elif isinstance(sample, SOMSample):
        return {"__somsample__": somsample_to_json(sample)}


def fcssample_to_json(sample: FCSSample) -> dict:
    sdict = asdict(sample)
    sdict["date"] = sample.date.isoformat()
    sdict["path"] = str(sample.path)
    if sample.material:
        sdict["material"] = sample.material.name
    else:
        sdict["material"] = ""
    return sdict


def somsample_to_json(sample: SOMSample) -> dict:
    sdict = asdict(sample)
    sdict["date"] = sample.date.isoformat()
    sdict["path"] = str(sample.path)
    return sdict


def json_to_sample(samplejson: dict) -> Sample:
    if "__fcssample__" in samplejson:
        return json_to_fcssample(samplejson["__fcssample__"])
    elif "__somsample__" in samplejson:
        return json_to_somsample(samplejson["__somsample__"])
    else:
        raise NotImplementedError("Unknown sample type")


def json_to_fcssample(samplejson: dict) -> FCSSample:
    samplejson["date"] = utils.str_to_date(samplejson["date"])
    samplejson["path"] = utils.URLPath(samplejson["path"])
    if samplejson["material"]:
        samplejson["material"] = mappings.Material[samplejson["material"]]
    else:
        samplejson["material"] = None
    return FCSSample(**samplejson)


def json_to_somsample(samplejson: dict) -> SOMSample:
    samplejson["date"] = utils.str_to_date(samplejson["date"])
    samplejson["path"] = utils.URLPath(samplejson["path"])
    samplejson["dims"] = tuple(samplejson["dims"])
    return SOMSample(**samplejson)


@with_slots
@dataclass
class Sample:
    """Single sample from a certain tube."""
    id: str
    case_id: str
    date: date
    tube: str
    path: utils.URLPath = None
    markers: List[str] = field(default_factory=list)

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
    panel: str = None
    count: int = None
    material: mappings.Material = None

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
    original_id: Union[str, tuple] = None  # sample id of original fcs used to build som
    dims: Tuple[int, int, int] = None

    def get_data(self) -> som.SOM:
        """
        Args:
            normalized: Normalize data to mean and standard deviation.
            scaled: Scale data between 0 and 1.
        Returns:
            Dataframe with fcs data.
        """
        data = som.SOM.from_path(self.path)
        data.material = self.material
        data.tube = self.tube
        data.cases = [self.id]
        return data
