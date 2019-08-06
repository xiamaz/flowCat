from __future__ import annotations
from typing import Iterable, List, Generator, Dict, Union
from flowcat.mappings import Material
from flowcat.dataset import case, case_dataset, fcs
from flowcat import utils
from .base import SOMCollection, SOM
from .fcssom import FCSSom


class CaseSomSampleException(Exception):
    def __init__(self, case_id, *args):
        self.case_id = case_id
        self.args = args

    def __str__(self):
        return f"Unable to obtain sample from case {self.case_id} with params: {self.args}"


class CaseSingleSom:
    """Transform single tube for a case to SOM."""

    config_name = "casesinglesom_config.json"

    def __init__(
            self,
            tube: str,
            materials: List[Material] = None,
            train_labels: List[str] = None,
            model: FCSSom = None,
            *args, **kwargs):
        self.tube = tube
        self.materials = materials
        self.train_labels = train_labels or []
        self.model = model or FCSSom(*args, **kwargs)

    @classmethod
    def load(cls, path: utils.URLPath, **kwargs):
        config = utils.load_json(path / cls.config_name)
        model = FCSSom.load(path, **kwargs)
        return cls(model=model, **config)

    @property
    def config(self):
        return {
            "tube": self.tube,
            "materials": self.materials,
            "train_labels": self.train_labels,
        }

    @property
    def weights(self):
        mweights = self.model.weights
        mweights.tube = self.tube
        mweights.cases = self.train_labels
        return mweights

    def save(self, path):
        path = utils.URLPath(path)
        self.model.save(path)
        utils.save_json(self.config, path / self.config_name)

    def train(self, data: Iterable[case.Case], *args, **kwargs) -> CaseSingleSom:
        tsamples = [c.get_tube(self.tube, materials=self.materials).data for c in data]
        self.model.train(tsamples)
        self.train_labels = [c.id for c in data]
        return self

    def transform(self, data: case.Case, *args, **kwargs) -> SOM:
        tsample = data.get_tube(self.tube, materials=self.materials)
        if tsample is None:
            raise CaseSomSampleException(data.id, self.tube, self.materials)
        somdata = self.model.transform(tsample.data, *args, **kwargs)
        somdata.cases = [data.id]
        somdata.tube = tsample.tube
        somdata.material = tsample.material
        return somdata

    def transform_generator(self, data: Iterable[case.Case], *args, **kwargs) -> Generator[SOM]:
        for casedata in data:
            yield casedata, self.transform(casedata, *args, **kwargs)


class CaseSom:
    """Create a SOM for a single case."""

    def __init__(
            self,
            tubes: Dict[str, list] = None,
            modelargs: dict = None,
            models: Dict[str, CaseSingleSom] = None,
            materials: list = None,
            tensorboard_dir: utils.URLPath = None,
    ):
        """
        Args:
            tubes: List of tube numbers or a dict mapping tube numbers to list of markers.
            modelargs: Dict of args to the CaseSingleSom class.
            models: Alternatively directly give a dictionary mapping tubes to SOM models.
            tensorboard_dir: Path for logging data. Each tube will be saved separately.
            materials: List of allowed materials, as enum of MATERIAL
        """
        self.materials = materials
        if models is None:
            self.models = {}
            for tube, markers in tubes.items():
                self.models[tube] = CaseSingleSom(
                    tube=tube,
                    markers=markers,
                    tensorboard_dir=tensorboard_dir / f"tube{tube}" if tensorboard_dir else None,
                    materials=materials,
                    **modelargs)
        else:
            self.models = models

    @classmethod
    def load(cls, path: utils.URLPath, tensorboard_dir: utils.URLPath = None, **kwargs):
        """Load a saved SOM model."""
        singlepaths = {p.name.lstrip("tube"): p for p in path.ls() if "tube" in str(p)}
        models = {}
        for tube, mpath in sorted(singlepaths.items()):
            tbdir = tensorboard_dir / f"tube{tube}" if tensorboard_dir else None
            models[tube] = CaseSingleSom.load(mpath, tensorboard_dir=tbdir, **kwargs)
        return cls(models=models)

    def save(self, path: utils.URLPath):
        """Save the model to the given directory"""
        for tube, model in self.models.items():
            output_path = path / f"tube{tube}"
            model.save(output_path)

    def train(self, data: Iterable[case.Case], *args, **kwargs) -> CaseSom:
        for tube, model in self.models.items():
            print(f"Training tube {tube}")
            model.train(data, *args, **kwargs)
        return self

    def transform(self, data: case.Case, *args, **kwargs) -> SOMCollection:
        casesoms = SOMCollection(cases=[data.id])
        for tube, model in self.models.items():
            print(f"Transforming tube {tube}")
            tsom = model.transform(data, *args, **kwargs)
            casesoms.add_som(tsom)
        return casesoms
