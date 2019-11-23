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


BMU_CALC = cmu_calculator(tf.Session())


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


@dataclass
class FlowCatPrediction:
    case: fc_case.Case
    som: fc_case.Case
    predictions: dict


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
    def load(cls, path: utils.URLPath = None, ref_path: utils.URLPath = None, cls_path: utils.URLPath = None):
        """Load classifier from the given path, alternatively give a separate path for reference and classifier."""
        if path is not None:
            ref_path = path / "reference"
            cls_path = path / "classifier"
        elif ref_path is None and cls_path is None:
            raise ValueError("Either path or ref_path and cls_path need to be set.")

        return cls(
            io_functions.load_casesom(ref_path),
            SOMClassifier.load(cls_path),
            SOMSaliency.load(cls_path)
        )

    def save(self, path: utils.URLPath):
        """Save the current model into the given path."""
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
        preds = self.classifier.predict([somcase])
        return FlowCatPrediction(case, somcase, preds)

    def generate_saliency(self, prediction: FlowCatPrediction, target_group):
        # returns eg 3x32x32 gradients
        gradients = self.saliency.transform(prediction.som, group=target_group, maximization=True)

        # map gradients to fcs by using run till tensor
        som_dict = []
        for gradient, (tube, model) in zip(gradients, self.reference.models.items()):
            somdata = prediction.som.get_tube(tube, kind="som").get_data()
            fcsdata, _ = prediction.case.get_tube(tube, kind="fcs").get_data()
            data, mask = model.model.prepare_data(fcsdata)
            mapped = BMU_CALC(somdata.data, fcsdata.data)
            fcsmapped = gradient[mapped]
            som_dict.append(SaliencyMapping(gradient, fcsmapped, tube))

        return som_dict
