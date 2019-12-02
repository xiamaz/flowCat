from typing import Iterable, List, Dict, Tuple
import datetime

import numpy as np

from flowcat import utils
from flowcat.mappings import Material
from flowcat.dataset import case as fc_case, sample as fc_sample
from flowcat.dataset.som import SOM

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
            tube: str,
            materials: List[Material] = None,
            train_labels: List[str] = None,
            run_identifier: str = None,
            model: FCSSom = None,
            *args, **kwargs):
        self.tube = tube
        self.materials = materials
        self.train_labels = train_labels or []
        self.model = model or FCSSom(*args, **kwargs)

        self.model_time = datetime.datetime.now().date()  # time of model

        if run_identifier is None:
            self.run_identifier = f"{self.model.name}_{utils.create_stamp()}"
        else:
            self.run_identifier = utils.create_stamp()

    @property
    def config(self):
        return {
            "tube": self.tube,
            "materials": self.materials,
            "train_labels": self.train_labels,
        }

    @property
    def weights(self):
        return self.model.weights

    def train(self, data: Iterable[fc_sample.FCSSample], *args, **kwargs) -> "CaseSingleSom":
        tsamples = [c.get_data() for c in data]
        self.model.train(tsamples)
        self.train_labels = [c.id for c in data]
        return self

    def calculate_nearest_nodes(self, data: fc_sample.FCSSample) -> np.array:
        return self.model.calculate_nearest_nodes(data.get_data())

    def transform(self, data: fc_sample.FCSSample, *args, **kwargs) -> fc_sample.SOMSample:
        if data is None:
            raise CaseSomSampleException(data.id, self.tube, self.materials)

        somdata = self.model.transform(data.get_data(), label=data.id, *args, **kwargs)
        som_id = f"{data.case_id}_t{self.tube}_{self.run_identifier}"
        somsample = fc_sample.SOMSample(
            id=som_id,
            case_id=data.case_id,
            original_id=data.id,
            date=self.model_time,
            tube=self.tube,
            dims=somdata.dims,
            markers=self.model.markers,
            data=somdata)
        return somsample

    def transform_generator(
            self,
            data: Iterable[fc_sample.FCSSample],
            *args, **kwargs
    ) -> Iterable[fc_sample.SOMSample]:
        for casedata in data:
            yield self.transform(casedata, *args, **kwargs)


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

    @property
    def tubes(self):
        return list(self.models.keys())

    def calculate_nearest_nodes(self, data: fc_case.Case) -> dict:
        return {
            tube: model.calculate_nearest_nodes(data.get_data(tube, kind="fcs"))
            for tube, model in self.models.items()
        }

    def train(self, data: Iterable[fc_case.Case], *args, **kwargs) -> "CaseSom":
        for tube, model in self.models.items():
            print(f"Training tube {tube}")
            model.train([d.get_tube(tube) for d in data], *args, **kwargs)
        return self

    def transform(self, data: fc_case.Case, *args, **kwargs) -> fc_case.Case:
        samples = []
        for tube, model in self.models.items():
            print(f"Transforming tube {tube}")
            somsample = model.transform(data.get_tube(tube, kind="fcs"), *args, **kwargs)
            samples.append(somsample)

        newcase = data.copy(samples=samples)
        return newcase

    def transform_generator(self, data: Iterable[fc_case.Case], **kwargs) -> Iterable[Tuple[fc_case.Case, fc_sample.SOMSample]]:
        for tube, model in self.models.items():
            for single in data:
                yield single, model.transform(single.get_tube(tube), **kwargs)
