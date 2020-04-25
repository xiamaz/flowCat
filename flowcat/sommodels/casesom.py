from typing import Iterable, List, Dict, Tuple
import datetime

import numpy as np

from flowcat import utils
from flowcat.types.material import Material
from flowcat.dataset import case as fc_case, sample as fc_sample
from flowcat.dataset.som import SOM
from flowcat.preprocessing.case_sample_merge import CaseSampleMergeTransformer

from .fcssom import FCSSom


class CaseSomSampleException(Exception):
    def __init__(self, case_id, *args):
        self.case_id = case_id
        self.args = args

    def __str__(self):
        return f"Unable to obtain sample from case {self.case_id} with params: {self.args}"


class CaseMergeSom:
    """Transform case into a single merged SOM."""

    def __init__(self, merger: "CaseSampleMergeTransformer", model: "FCSSom", run_identifier: str, model_time: "datetime" = None):
        self._merger = merger
        self._model = model
        if model_time is None:
            self.model_time = datetime.datetime.now()
        else:
            self.model_time = model_time
        self.run_identifier = run_identifier

    @property
    def config(self):
        return {
            "channels": self.markers,
            "run_identifier": self.run_identifier,
            "model_time": self.model_time,
        }

    @property
    def markers(self):
        return self._merger._channels

    @classmethod
    def load_from_config(cls, config: dict, model: "FCSSom"):
        merger = CaseSampleMergeTransformer(config["channels"])
        return cls(
            merger=merger,
            model=model,
            run_identifier=config["run_identifier"],
            model_time=config["model_time"])

    @classmethod
    def create(cls, channels: "List[Marker]", run_identifier: str, model: "FCSSom" = None, **kwargs):
        if model is None:
            kwargs["markers"] = channels
            model = FCSSom(**kwargs)
        return cls.load_from_config(
            {"channels": channels, "run_identifier": run_identifier, "model_time": None},
            model=model)

    def get_som_id(self, case: "Case") -> str:
        return f"{case.id}_{self.run_identifier}"

    def train(self, data: "Iterable[Case]"):
        fcsdatas = [self._merger.transform(c) for c in data]
        self._model.train(fcsdatas)
        return self

    def transform(self, data: "Case"):
        fcsdata = self._merger.transform(data)
        somdata = self._model.transform(fcsdata)
        som_id = self.get_som_id(data)
        somsample = fc_sample.SOMSample(
            id=som_id,
            case_id=data.id,
            original_id=data.id,
            date=self.model_time.date(),
            tube="",
            dims=somdata.dims,
            markers=self._model.markers,
            data=somdata)
        somcase = data.copy()
        somcase.samples = [somsample]
        return somcase

    def transform_generator(
            self,
            data: Iterable[fc_case.Case],
            *args, **kwargs
    ) -> Iterable[fc_sample.SOMSample]:
        for casedata in data:
            yield self.transform(casedata, *args, **kwargs)


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

    @property
    def som_config(self):
        return {
            tube: {
                "dims": m.model.dims,
                "channels": m.model.markers,
            } for tube, m in self.models.items()
        }

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
