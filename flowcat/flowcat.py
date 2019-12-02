"""
Major wrapper class for generating SOMs and prediction on existing classifier
and SOM data.
"""
import numpy as np
from dataclasses import dataclass

import tensorflow as tf

from flowcat import utils, io_functions
from flowcat.classifier import SOMClassifier, SOMSaliency
from flowcat.classifier.saliency import bmu_calculator
from flowcat.sommodels.casesom import CaseSom
from flowcat.dataset import case as fc_case, sample as fc_sample


BMU_CALC = bmu_calculator(tf.Session())


def case_from_dict(case_dict) -> fc_case.Case:
    """Use very simple dict to generate case objects usable for prediction.

    Format:
        {"id": case id, "samples": [list of paths to tubes in correct order]}
    """
    samples = [
        fc_sample.FCSSample(id=p.name, case_id=case_dict["id"], tube=i, path=p)
        for i, p in enumerate(map(utils.URLPath, case_dict["samples"]))
    ]
    case = fc_case.Case(
        id=case_dict["id"],
        samples=samples)
    return case


def load_case_collection(data: str, meta: str = None):
    if meta is None:
        meta = utils.URLPath(data) / "meta.json.gz"
        data = utils.URLPath(data) / "data"
    else:
        data = utils.URLPath(data)
        meta = utils.URLPath(meta)
    return io_functions.load_case_collection(data, meta)


@dataclass
class FlowCatPrediction:
    case: fc_case.Case
    som: fc_case.Case
    predictions: dict

    @property
    def predicted_group(self):
        return max(self.predictions.keys(), key=lambda key: self.predictions[key])


@dataclass
class SaliencyMapping:
    som: np.array
    fcs: np.array
    tube: str


class FlowCat:
    def __init__(self, reference: CaseSom = None, classifier: SOMClassifier = None, saliency: SOMSaliency = None):
        """Initialization with optional existing models."""
        self.reference = reference
        self.classifier = classifier
        self.saliency = saliency

    @classmethod
    def load(cls, path: str = None, ref_path: str = None, cls_path: str = None):
        """Load classifier from the given path, alternatively give a separate path for reference and classifier."""
        if path is not None:
            ref_path = utils.URLPath(path) / "reference"
            cls_path = utils.URLPath(path) / "classifier"
        elif ref_path is not None and cls_path is not None:
            ref_path = utils.URLPath(ref_path)
            cls_path = utils.URLPath(cls_path)
        else:
            raise ValueError("Either path or ref_path and cls_path need to be set.")

        return cls(
            io_functions.load_casesom(ref_path),
            SOMClassifier.load(cls_path),
            SOMSaliency.load(cls_path)
        )

    def save(self, path: str):
        """Save the current model into the given path."""
        path = utils.URLPath(path)
        path.mkdir()
        io_functions.save_casesom(self.reference, path / "reference")
        self.classifier.save(path / path / "classifier")

    def predict_dict(self, case_dict: dict) -> dict:
        """Create dict predictions."""
        case = case_from_dict(case_dict)
        prediction = self.predict(case)
        return prediction.predictions

    def predict(self, case: fc_case.Case) -> FlowCatPrediction:
        somcase = self.reference.transform(case)
        pred = self.classifier.predict([somcase])[0]
        return FlowCatPrediction(case, somcase, pred)

    def generate_saliency(self, prediction: FlowCatPrediction, target_group):
        # returns eg 3x32x32 gradients
        gradients = self.saliency.transform(prediction.som, group=target_group, maximization=True)

        # map gradients to fcs by using run till tensor
        som_dict = []
        for gradient, (tube, model) in zip(gradients, self.reference.models.items()):
            somdata = prediction.som.get_tube(tube, kind="som").get_data()
            fcsdata = prediction.case.get_tube(tube, kind="fcs").get_data()
            data, _ = model.model.prepare_data(fcsdata)
            mapped = BMU_CALC(somdata.data.reshape((-1, data.shape[-1])), data)
            gradient = gradient.reshape((-1,))
            fcsmapped = gradient[mapped]
            som_dict.append(SaliencyMapping(gradient, fcsmapped, tube))

        return som_dict
