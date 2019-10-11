from typing import Iterable, List, Dict, Tuple
import datetime

from flowcat import utils
from flowcat.mappings import Material
from flowcat.dataset import case, sample
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

    def train(self, data: Iterable[case.Case], *args, **kwargs) -> "CaseSingleSom":
        tsamples = [c.get_tube(self.tube).get_data() for c in data]
        self.model.train(tsamples)
        self.train_labels = [c.id for c in data]
        return self

    def transform(self, data: case.Case, *args, **kwargs) -> sample.SOMSample:
        fcs_sample = data.get_tube(self.tube)

        if fcs_sample is None:
            raise CaseSomSampleException(data.id, self.tube, self.materials)

        somdata = self.model.transform(fcs_sample.get_data(), label=data.id, *args, **kwargs)
        som_id = f"{data.id}_t{self.tube}_{self.run_identifier}"
        somsample = sample.SOMSample(
            id=som_id,
            case_id=data.id,
            original_id=fcs_sample.id,
            date=self.model_time,
            tube=self.tube,
            dims=somdata.dims,
            markers=self.model.markers,
            data=somdata)
        return somsample

    def transform_generator(
            self,
            data: Iterable[case.Case],
            *args, **kwargs
    ) -> Iterable[Tuple[case.Case, sample.SOMSample]]:
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

    @property
    def tubes(self):
        return list(self.models.keys())

    def train(self, data: Iterable[case.Case], *args, **kwargs) -> "CaseSom":
        for tube, model in self.models.items():
            print(f"Training tube {tube}")
            model.train(data, *args, **kwargs)
        return self

    def transform(self, data: case.Case, *args, **kwargs) -> case.Case:
        samples = []
        for tube, model in self.models.items():
            print(f"Transforming tube {tube}")
            somsample = model.transform(data, *args, **kwargs)
            samples.append(somsample)

        newcase = data.copy(samples=samples)
        return newcase

    def transform_generator(self, data: Iterable[case.Case], **kwargs) -> Iterable[Tuple[case.Case, sample.SOMSample]]:
        for tube, model in self.models.items():
            for single in data:
                yield single, model.transform(single, **kwargs)
