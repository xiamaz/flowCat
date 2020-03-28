"""
Major wrapper class for generating SOMs and prediction on existing classifier
and SOM data.
"""
import logging
from shutil import rmtree
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

import tensorflow as tf

from flowcat import utils, io_functions, constants
from flowcat.classifier import SOMClassifier, SOMSaliency, SOMClassifierConfig, create_model_multi_input
from flowcat.classifier.predictions import generate_all_metrics
from flowcat.classifier.saliency import bmu_calculator
from flowcat.sommodels.casesom import CaseSom
from flowcat.dataset import case as fc_case, sample as fc_sample, case_dataset


TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 128


BMU_CALC = bmu_calculator(tf.Session())

LOGGER = logging.getLogger(__name__)


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
    data = utils.URLPath(data)
    if meta is not None:
        meta = utils.URLPath(meta)

    return io_functions.load_case_collection(data, meta)


def load_existing_casesom(path: utils.URLPath, args: dict) -> CaseSom:
    try:
        casesom = io_functions.load_casesom(path, **args)
        return casesom
    except Exception as e:
        LOGGER.warning("Failed to load existing casesom at %s with error %s:", path, e)
        return None


def load_matching_som_dataset(fcs_dataset: "CaseCollection", som_dataset_path: utils.URLPath) -> "CaseCollection":
    """Check whether the given som path contains a complete SOM dataset matching the given FCS dataset.

    Otherwise return None.
    """
    try:
        som_dataset = io_functions.load_case_collection(som_dataset_path)
    except Exception as e:
        LOGGER.warning("Loading existing dataset at %s produced error: %s", som_dataset_path, e)
        return None

    same_case_number = len(fcs_dataset) == len(som_dataset)
    same_sample_count = len([1 for c in fcs_dataset for s in c.samples]) == len([1 for c in som_dataset for s in c.samples])
    if not (same_case_number and same_sample_count):
        LOGGER.warning("Existing som dataset at %s does not match number of samples or cases of given FCS dataset", som_dataset_path)
        return None
    return som_dataset


def transform_dataset_to_som(som_reference: CaseSom, dataset: "CaseCollection", output: utils.URLPath):
    """Transform dataset into som dataste using the given reference SOM model.
    """
    print(f"Trainsforming individual samples")
    data_output = output / "data"
    meta_output = output / "meta.json.gz"
    config_output = output / "config.json"

    data_output.mkdir()

    casesamples = defaultdict(list)
    count_samples = len(dataset) * len(som_reference.models)
    countlen = len(str(count_samples))
    for i, (case, somsample) in enumerate(utils.time_generator_logger(som_reference.transform_generator(dataset))):
        sompath = data_output / f"{case.id}_t{somsample.tube}.npy"
        io_functions.save_som(somsample.data, sompath, save_config=False)
        somsample.data = None
        somsample.path = sompath.relative_to(data_output)
        print(type(somsample.path), somsample.path)
        casesamples[case.id].append(somsample)
        print(f"[{str(i + 1).rjust(countlen, ' ')}/{count_samples}] Created tube {somsample.tube} for {case.id}")

    print(f"Saving result to new collection at {output}")
    som_dataset = case_dataset.CaseCollection([
        case.copy(samples=casesamples[case.id])
        for case in dataset
    ])
    som_dataset.selected_markers = {
        m.tube: m.model.markers for m in som_reference.models.values()
    }
    io_functions.save_case_collection(som_dataset, meta_output)
    io_functions.save_json(som_reference.som_config, config_output)
    return som_dataset


def reconfigure_som_model(som_model: CaseSom, args: dict) -> CaseSom:
    """Reconfigure SOM by saving a copy and loading it again."""
    tmp_path = utils.URLPath("/tmp/flowcat/sommodel")

    io_functions.save_casesom(som_model, tmp_path)
    reconfigured_model = io_functions.load_casesom(tmp_path, **args)

    rmtree(str(tmp_path))

    return reconfigured_model


def check_dataset_groups(dataset, groups):
    """Check that all groups given are actually contained in the dataset.
    """
    dataset_groups = {d.group for d in dataset}
    return set(groups) == dataset_groups


def prepare_classifier_train_dataset(
        dataset: "CaseCollection",
        split_ratio=0.9,
        mapping=None,
        groups=None,
        balance=None, val_dataset: "CaseCollection" = None):
    """Prepare dataset splitting and optional upsampling.

    Args:
        split_ratio: Ratio of training set to total dataset.
        mapping: Optionally map existing groups to new groups contained in mapping.
        groups: List of groups to be used.
        balance: Dict or value to upsample the training dataset.
    """
    if mapping:
        dataset = dataset.map_groups(mapping)

    if groups:
        dataset = dataset.filter(groups=groups)
        if not check_dataset_groups(dataset, groups):
            raise RuntimeError(f"Group mismatch: Not all groups in {groups} are in dataset.")

    if val_dataset is not None:
        LOGGER.info("Received existing validation data, will not split.")
        train, validate = dataset, val_dataset
    elif split_ratio < 1.0:
        train, validate = dataset.create_split(split_ratio, stratify=True)
    else:
        train, validate = dataset, None

    if balance:
        train = train.balance_per_group(balance)
    return train, validate


def train_som_classifier(
    train_dataset: "CaseCollection",
    validate_dataset: "CaseCollection",
    config: SOMClassifierConfig = None,
) -> "SOMClassifier":
    """Configure the dataset based on config and train a given model."""
    model = SOMClassifier(config)
    model.create_model(create_model_multi_input)

    train = model.create_sequence(train_dataset, config.train_batch_size)

    if validate_dataset is not None:
        validate = model.create_sequence(validate_dataset, config.valid_batch_size)
    else:
        validate = None

    model.train_generator(train, validate, epochs=config.train_epochs, class_weight=None)
    return model


def generate_prediction_metrics(predictions: "List[FlowCatPrediction]", mapping: dict, output: utils.URLPath) -> dict:
    true_labels = [p.case.group for p in predictions]
    pred_labels = [p.predicted_group for p in predictions]

    confusion, metrics = generate_all_metrics(true_labels, pred_labels, mapping, output)
    return {
        "confusion": confusion,
        "metrics": metrics
    }


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

    def _train_reference(self, reference: "Iterable[Case]", args: dict, transform_args: dict, output: utils.URLPath, tensorboard_dir: utils.URLPath = None) -> "CaseSom":
        """Train the reference SOM or load preexisting."""
        sommodel = load_existing_casesom(output, args)
        if sommodel is not None:
            LOGGER.info("Loaded existing model at %s", output)
            self.reference = reconfigure_som_model(sommodel, transform_args)
            return self.reference

        selected_markers = reference.selected_markers
        sommodel = CaseSom(
            tubes=selected_markers,
            tensorboard_dir=tensorboard_dir,
            modelargs=args)
        sommodel.train(reference)

        io_functions.save_casesom(sommodel, output)

        self.reference = reconfigure_som_model(sommodel, transform_args)
        return self.reference

    def _transform_dataset(self, dataset: "CaseCollection", output: utils.URLPath) -> "CaseCollection":
        # try to load an existing SOM dataset
        existing_dataset = load_matching_som_dataset(dataset, output)
        if existing_dataset is not None:
            LOGGER.info("Using existing SOM dataset at: %s", output)
            return existing_dataset

        som_dataset = transform_dataset_to_som(self.reference, dataset, output)
        return som_dataset

    def _adapt_classifier_config(self, config: "SOMClassifierConfig") -> "SOMClassifierConfig":
        new_config = config.copy()
        new_config.tubes = self.reference.som_config
        return new_config

    def _train_classifier(self, dataset: "CaseCollection", args: dict, val_dataset: "CaseCollection" = None) -> "SOMClassifier":
        config = self._adapt_classifier_config(args["config"])
        train, validate = prepare_classifier_train_dataset(
            dataset,
            split_ratio=args["split_ratio"],
            groups=config.groups,
            balance=args["balance"],
            mapping=config.mapping, val_dataset=val_dataset)
        self.classifier = train_som_classifier(train, validate, config)
        # return self.classifier
        return validate

    def train(self, dataset: "CaseCollection", reference: "List[Case]", output: utils.URLPath, validation_data: "CaseCollection" = None, args: dict = None):
        """Train a new model using the given dataset."""
        if self.reference != None or self.classifier != None:
            raise RuntimeError("flowCat model has already been trained")

        if args is None:
            args = constants.DEFAULT_TRAIN_ARGS

        ref_output = output / "reference"
        som_output = output / "som"
        som_output_val = output / "som_val"

        self._train_reference(reference, args["reference"], args["transform"], ref_output)
        som_dataset = self._transform_dataset(dataset, som_output)
        val_som_dataset = self._transform_dataset(validation_data, som_output_val)
        self._train_classifier(som_dataset, args=args["classifier"], val_dataset=val_som_dataset)
        return som_dataset, val_som_dataset

    def predict_dict(self, case_dict: dict) -> dict:
        """Create dict predictions."""
        case = case_from_dict(case_dict)
        prediction = self.predict(case)
        return prediction.predictions

    def predict_dataset(self, dataset: "Iterable[Case]") -> "List[FlowCatPrediction]":
        return [self.predict(c) for c in dataset]

    def predict(self, case: fc_case.Case) -> FlowCatPrediction:
        if "som" not in case.sample_kinds:
            somcase = self.reference.transform(case)
        else:
            somcase = case
        pred = self.classifier.predict([somcase])[0]
        return FlowCatPrediction(case, somcase, pred)

    def predictions_to_metric(self, predictions: "List[FlowCatPrediction]", output: utils.URLPath) -> dict:
        mapping = {
            "groups": self.classifier.config.groups,
            "map": {},
        }
        return generate_prediction_metrics(predictions, mapping, output)

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
