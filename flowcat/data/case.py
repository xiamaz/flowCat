import os
import logging
import enum
import functools
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

import fcsparser

from ..utils import URLPath


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


class Case:
    """Basic case object containing all metadata for a case."""
    __slots__ = (
        "_json",
        "path",
        "_filepaths",
        "used_material",
        "date",
        "infiltration",
        "short_diagnosis",
        "sureness_description",
        "sureness",
        "group",
        "id",
    )

    def __init__(self, data, path=""):
        """
        Args:
            data: Contains all metainformation, either a dictionary or
                a case object
            path: Path prefix used for loading any data.
        Returns:
            New case object.
        """
        self._filepaths = None
        self.used_material = None

        if isinstance(data, self.__class__):
            self._json = data._json.copy()
            self.path = data.path
            self.date = data.date
            self.infiltration = data.infiltration
            self.group = data.group
            self.id = data.id
            self.short_diagnosis = data.short_diagnosis
            self.sureness_description = data.sureness_description
            self.sureness = data.sureness
            self.filepaths = data.filepaths
        else:
            self._json = data
            self.path = path

            self.date = datetime.strptime(data["date"], "%Y-%m-%d").date()
            self.infiltration = data["infiltration"]
            self.group = data["cohort"]
            self.id = data["id"]
            self.short_diagnosis = data["diagnosis"]
            self.sureness_description = data["sureness"]
            self.sureness = self._infer_sureness()

            self.filepaths = data["filepaths"]

    @property
    def json(self):
        cdict = {
            "id": self.id,
            "date": self.date.isoformat(),
            "filepaths": [p.json for p in self.filepaths],
            "cohort": self.group,
            "infiltration": self.infiltration,
            "diagnosis": self.short_diagnosis,
            "sureness": self.sureness_description,
        }
        return cdict

    @property
    def filepaths(self):
        """Get a list of filepaths."""
        return self._filepaths

    @filepaths.setter
    def filepaths(self, value: list):
        """Set filepaths and clear all generated dicts on data.
        Args:
            value: List of dictionaries with information to create case paths or case paths.
        """
        self._filepaths = [TubeSample(v, self) for v in value]

    def get_tube(self, tube, min_count=0, material=None):
        """Get the TubePath fulfilling the given requirements, return the
        last on the list if multiple are available.
        Args:
            tube: Int tube number to be selected.
            min_count: Minimum number of events in the FCS file.
            material: Type of material used for the tube.
        Returns:
            TubePath or None.
        """
        if self.used_material and not material:
            material = self.used_material

        tubecases = [
            p for p in self.filepaths
            if p.tube == tube and (
                material is None or material == p.material
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

    def has_selected_markers(self, selected_markers):
        """Check whether default samples can fulfill the selected markers."""
        for tube, markers in selected_markers.items():
            tcase = self.get_tube(tube)
            if not tcase.has_markers(markers):
                return False
        return True


    def get_possible_material(self, tubes, allowed_materials=None):
        available_materials = self.get_same_materials(tubes)
        if allowed_materials:
            filtered = [m for m in available_materials if m in allowed_materials]
        else:
            filtered = available_materials
        return filtered[0] if filtered else None

    def has_same_material(self, tubes, allowed_materials=None):
        if allowed_materials is None:
            return bool(self.get_same_materials(tubes))
        else:
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

    def copy(self):
        return self.__class__(self)


class TubeSample:
    __slots__ = (
        "count",
        "date",
        "path",
        "markers",
        "material",
        "material_desc",
        "parent",
        "panel",
        "result",
        "result_success",
        "tube",
    )

    """Single sample from a certain tube."""
    def __init__(self, path, parent):
        if isinstance(path, self.__class__):
            self.tube = path.tube
            self.material_desc = path.material_desc
            self.material = path.material
            self.path = path.path
            self.panel = path.panel
            self.markers = path.markers.copy()
            self.count = path.count
            self.date = path.date
        else:
            self.tube = int(path["tube"])
            self.material_desc = path["material"]
            self.material = Material.from_str(path["material"])
            self.panel = path["panel"]
            self.date = datetime.strptime(path["date"], "%Y-%m-%d").date()

            self.path = path["fcs"]["path"]
            self.markers = path["fcs"]["markers"]
            self.count = path["fcs"]["event_count"]

        self.parent = parent

    @property
    def json(self):
        return {
            "id": self.parent.id,
            "panel": self.panel,
            "tube": self.tube,
            "date": self.date.isoformat(),
            "material": self.material_desc,
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
        return self.get_data(normalized=False, scaled=False)

    @property
    def urlpath(self):
        url_path = URLPath(self.parent.path) / self.path
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
        url_path = URLPath(self.parent.path) / self.path
        data = FCSData(url_path.get())

        if normalized:
            data = FCSStandardScaler().fit_transform(data)
        if scaled:
            data = FCSMinMaxScaler().fit_transform(data)

        return data

    def has_markers(self, markers: list) -> bool:
        """Return whether given list of markers are fulfilled."""
        return all_in(markers, self.markers)


class FCSData:
    __slots__ = (
        "_meta", "data", "ranges"
    )

    default_encoding = "latin-1"
    default_dataset = 0

    def __init__(self, initdata):
        """Create a new FCS object.

        Args:
            initdata: Either tuple of meta and data from fcsparser, string filepath or another FCSData object.
        Returns:
            FCSData object.
        """
        if isinstance(initdata, self.__class__):
            self._meta = initdata.meta.copy()
            self.ranges = initdata.ranges.copy()
            self.data = initdata.data.copy()
        else:
            # unpack metadata, data tuple
            if isinstance(initdata, tuple):
                meta, data = initdata
            # load using filepath
            else:
                meta, data = fcsparser.parse(
                    str(initdata),
                    data_set=self.default_dataset,
                    encoding=self.default_encoding)
            self._meta = meta
            self.data = data
            self.ranges = self._get_ranges_from_pnr(self._meta)

    @property
    def meta(self):
        return self._meta

    def copy(self):
        return self.__class__(self)

    def drop_empty(self):
        """Drop all channels containing nix in the channel name.
        """
        nix_cols = [c for c in self.data.columns if "nix" in c]
        self.drop_channels(nix_cols)
        return self

    def drop_channels(self, channels):
        """Drop the given columns from the data.
        Args:
            channels: List of channels or channel name to drop. Will not throw an error if the name is not found.
        Returns:
            self. This operation is done in place, so the original object will be modified!
        """
        self.data.drop(channels, axis=1, inplace=True, errors="ignore")
        self.ranges.drop(channels, axis=1, inplace=True, errors="ignore")
        return self

    def _get_ranges_from_pnr(self, metadata):
        """Get ranges from metainformation."""
        pnr = {
            c: {
                "min": 0,
                "max": int(metadata[f"$P{i + 1}R"])
            } for i, c in enumerate(self.data.columns)
        }
        pnr = pd.DataFrame.from_dict(pnr, orient="columns", dtype="float32")
        return pnr

    def __repr__(self):
        """Print string representation of the input file."""
        nevents, nchannels = self.data.shape
        return f"<FCS :: {nevents} events :: {nchannels} channels>"


class FCSPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, *_):
        for _, model in self.steps:
            model.fit(X)
        return self

    def transform(self, X, *_):
        for _, model in self.steps:
            X = model.transform(X)
        return X

    def fit_transform(self, X, *_):
        return self.fit(X).transform(X)


class FCSMarkersTransform(TransformerMixin, BaseEstimator):
    def __init__(self, markers):
        self._markers = markers

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        if isinstance(X, FCSData):
            X.data = X.data.loc[:, self._markers]
        else:
            X = X.loc[:, self._markers]
        return X


class FCSLogTransform(BaseEstimator, TransformerMixin):
    """Transform FCS files logarithmically.  Currently this does not work
    correctly, since FCS files are not $PnE transformed on import"""

    def transform(self, X, *_):
        names = [n for n in X.columns if "LIN" not in n]
        X[names] = np.log1p(X[names])
        return X

    def fit(self, *_):
        return self


class FCSScatterFilter(BaseEstimator, TransformerMixin):
    """Remove events with values below threshold in specified channels."""

    def __init__(
            self,
            filters=[("SS INT LIN", 0), ("FS INT LIN", 0)],
    ):
        self._filters = filters

    def transform(self, X, *_):
        if isinstance(X, FCSData):
            selected = functools.reduce(
                lambda x, y: x & y, [X.data[c] > t for c, t in self._filters])
            X.data = X.data.loc[selected, :]
        else:
            selected = functools.reduce(
                lambda x, y: x & y, [X[c] > t for c, t in self._filters])
            X = X.loc[selected, :]
        return X

    def fit(self, *_):
        return self


class FCSMinMaxScaler(TransformerMixin, BaseEstimator):
    """MinMaxScaling with adaptations for FCSData."""

    def __init__(self):
        self._model = preprocessing.MinMaxScaler()

    def fit(self, X, *_):
        if isinstance(X, FCSData):
            data = X.ranges
        else:
            data = X.data
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        if isinstance(X, FCSData):
            data = self._model.transform(X.data)
            X.data = pd.DataFrame(data, columns=X.data.columns, index=X.data.index)
            ranges = self._model.transform(X.ranges)
            X.ranges = pd.DataFrame(ranges, columns=X.ranges.columns, index=X.ranges.index)
        elif isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                self._model.transform(X),
                columns=X.columns, index=X.index)
        else:
            X = self._model.transform(X)
        return X


class FCSStandardScaler(TransformerMixin, BaseEstimator):
    """Standard deviation scaling adapted for FCSData objects."""
    def __init__(self):
        self._model = preprocessing.StandardScaler()

    def fit(self, X, *_):
        if isinstance(X, FCSData):
            data = X.data
        else:
            data = X
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        if isinstance(X, FCSData):
            data = self._model.transform(X.data)
            ranges = self._model.transform(X.ranges)
            X.data = pd.DataFrame(data, columns=X.data.columns, index=X.data.index)
            X.ranges = pd.DataFrame(ranges, columns=X.ranges.columns, index=X.ranges.index)
        elif isinstance(X, pd.DataFrame):
            data = self._model.transform(X)
            X = pd.DataFrame(data, columns=X.columns, index=X.index)
        else:
            X = self._model.transform(X)
        return X
