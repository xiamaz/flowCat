from __future__ import annotations
from typing import Iterable, List, Generator
from flowcat.mappings import Material
from flowcat.dataset import case, case_dataset, fcs
from .fcssom import FCSSom


class CaseSomSampleException(Exception):
    def __init__(self, case_id, *args):
        self.case_id = case_id
        self.args = args

    def __str__(self):
        return f"Unable to obtain sample from case {self.case_id} with params: {self.args}"


class CaseSingleSom:
    """Transform single tube for a case to SOM."""

    def __init__(
            self,
            tube: int,
            materials: List[Material],
            *args, **kwargs):
        self.tube = tube
        self.materials = materials
        self.model = FCSSom(*args, **kwargs)
        self.train_labels = []

    @property
    def weights(self):
        mweights = self.model.weights
        mweights.tube = self.tube
        mweights.cases = self.train_labels
        return mweights

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
