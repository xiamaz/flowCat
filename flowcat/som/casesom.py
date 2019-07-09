from __future__ import annotations
from typing import Iterable, List, Generator
from flowcat.mappings import Material
from flowcat.dataset import case, case_dataset, fcs
from flowcat import utils
from .base import SOMCollection
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
            tube: int,
            materials: List[Material],
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

    def train(self, data: Iterable[case.Case], *args, **kwargs):
        tsamples = [c.get_tube(self.tube, materials=self.materials).data for c in data]
        self.model.train(tsamples)
        self.train_labels = [c.id for c in data]
        return self

    def transform(self, data: case.Case, *args, **kwargs):
        tsample = data.get_tube(self.tube, materials=self.materials)
        if tsample is None:
            raise CaseSomSampleException(data.id, self.tube, self.materials)
        somdata = self.model.transform(tsample.data, *args, **kwargs)
        somdata.cases = [data.id]
        somdata.tube = tsample.tube
        somdata.material = tsample.material
        return somdata

    def transform_generator(self, data: Iterable[case.Case], *args, **kwargs):
        for casedata in data:
            yield self.transform(casedata, *args, **kwargs)


class CaseSom:
    def __init__(self, tubes, materials, modelargs):
        self.tubes = tubes
        self.materials = materials
        self.models = {
            t: CaseSingleSom(tube=t, materials=materials, **modelargs)
            for t in self.tubes
        }

    def train(self, data: Iterable[case.Case], *args, **kwargs):
        for tube in self.tubes:
            print(f"Training tube {tube}")
            self.models[tube].train(data, *args, **kwargs)

    def transform(self, data: case.Case, *args, **kwargs):
        casesoms = SOMCollection(case=data.id)
        for tube in self.tubes:
            print(f"Transforming tube {tube}")
            tsom = self.models[tube].transform(data, *args, **kwargs)
            casesoms.add_som(tsom)
        return casesoms
