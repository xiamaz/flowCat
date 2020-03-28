from copy import deepcopy
from dataclasses import dataclass, asdict, replace

import numpy as np
import keras
from sklearn.preprocessing import LabelBinarizer

from flowcat import io_functions, utils, som_dataset, classification_utils
from flowcat.dataset import case_dataset
from flowcat.plots import history as plot_history


def plot_training_history(history, output):
    history_data = {
        "accuracy": history.history["acc"],
        "val_accuracy": history.history["val_acc"],
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
    }
    acc_plot = plot_history.plot_history(history_data, title="Training accuracy")
    acc_plot.tight_layout()
    acc_plot.savefig(str(output), dpi=300)


@dataclass
class SOMClassifierConfig:
    """Configuration information usable by SOM classifier."""
    tubes: dict
    groups: list
    pad_width: int = 2
    mapping: dict = None
    cost_matrix: str = None
    train_epochs: int = 20
    train_batch_size: int = 32
    valid_batch_size: int = 128

    def to_json(self):
        return asdict(self)

    def copy(self):
        new_tubes = deepcopy(self.tubes)
        new_mapping = deepcopy(self.mapping)
        return replace(self, tubes=new_tubes, mapping=new_mapping)

    @property
    def output(self):
        return len(self.groups)

    @property
    def inputs(self):
        inputs = tuple(
            [*[dd + 2 * self.pad_width for dd in d["dims"][:-1]], len(d["channels"])] for d in self.tubes.values())
        return inputs

    def get_loss(self, modeldir=None):
        if self.cost_matrix is None:
            return "categorical_crossentropy"
        cost_matrix = np.load(modeldir / self.cost_matrix)
        return classification_utils.WeightedCategoricalCrossentropy(cost_matrix)


def load_somclassifier_config(path: utils.URLPath) -> SOMClassifierConfig:
    """Load somclassifier config from the given path."""
    return SOMClassifierConfig(**io_functions.load_json(path))


def save_somclassifier_config(config: SOMClassifierConfig, path: utils.URLPath):
    """Save configuration to the given path."""
    io_functions.save_json(config.to_json(), path)


def getter_case(data, tube):
    return data.get_tube(tube, kind="som").get_data().data


def getter_som(data, tube):
    return data.get_tube(tube, kind="som")


class SOMClassifier:
    def __init__(
            self,
            config: SOMClassifierConfig,
            model=None,
            binarizer: LabelBinarizer = None,
            modeldir: utils.URLPath = None,
            data_ids: dict = None):
        self.model = model
        self.config = config
        if binarizer is None:
            self.binarizer = LabelBinarizer().fit(self.config.groups)
        else:
            self.binarizer = binarizer

        self.data_ids = data_ids
        self.modeldir = modeldir

        self.training_history = []

    @classmethod
    def load(cls, path: utils.URLPath):
        """Load classifier model from the given path."""
        config = load_somclassifier_config(path / "config.json")
        model = keras.models.load_model(str(path / "model.h5"))
        binarizer = io_functions.load_joblib(path / "binarizer.joblib")

        data_ids = {
            "validation": io_functions.load_json(path / "ids_validate.json"),
            "train": io_functions.load_json(path / "ids_train.json"),
        }
        return cls(config, binarizer=binarizer, model=model, data_ids=data_ids, modeldir=path)

    def create_model(self, fun, kwargs=None, compile=True):
        """Create a model using the given model function."""
        if kwargs is None:
            kwargs = {}

        self.model = fun(self.config.inputs, self.config.output, **kwargs)
        if compile:
            self.model.compile(
                loss=self.config.get_loss(self.modeldir),
                optimizer="adam", metrics=["accuracy"])

    def save(self, path: utils.URLPath):
        """Save the given classifier model to the given path."""
        save_somclassifier_config(self.config, path / "config.json")
        self.model.save(str(path / "model.h5"))
        io_functions.save_joblib(self.binarizer, path / "binarizer.joblib")

        io_functions.save_json(self.data_ids["validation"], path / "ids_validate.json")
        io_functions.save_json(self.data_ids["train"], path / "ids_train.json")

    def save_information(self, path: utils.URLPath):
        """Save additional plots and information."""
        # Text summary of model
        with (path / "model_summary.txt").open("w") as summary_file:
            def print_file(*args, **kwargs):
                print(*args, **kwargs, file=summary_file)
            self.model.summary(print_fn=print_file)

        # Image plotting structure of model
        keras.utils.plot_model(self.model, to_file=str(path / "model_plot.png"))

        # plot all training history
        for i, (meta, history) in enumerate(self.training_history):
            training_output = path / f"train_{i}"
            io_functions.save_json(meta, training_output / "info.json")
            plot_training_history(history, training_output / "training.png")

    def get_validation_data(self, dataset: som_dataset.SOMDataset) -> som_dataset.SOMDataset:
        return dataset.filter(labels=self.data_ids["validation"])

    def create_sequence(
            self,
            dataset: som_dataset.SOMDataset,
            batch_size: int = 128,
            getter: "Callback" = None,
    ) -> som_dataset.SOMSequence:
        if getter is None:
            if isinstance(dataset, som_dataset.SOMDataset):
                getter = getter_som
            elif isinstance(dataset, case_dataset.CaseCollection):
                getter = getter_case
            else:
                raise ValueError(f"Unknown dataset type {type(dataset)} with no given getter.")

        seq = som_dataset.SOMSequence(
            dataset, self.binarizer,
            get_array_fun=getter,
            tube=self.config.tubes,
            batch_size=batch_size,
            pad_width=self.config.pad_width,
        )
        return seq

    def array_from_cases(self, cases):
        """Transform som data in a single into a format usable for prediction."""
        xdata = [
            np.array([
                som_dataset.pad_array(
                    case.get_tube(tube, kind="som").get_data().data,
                    self.config.pad_width)
                for case in cases
            ])
            for tube in self.config.tubes
        ]

        ydata = self.binarizer.transform([case.group for case in cases])
        return xdata, ydata

    def train_generator(self, train, validation=None, epochs=20, class_weight=None):
        """Train the current model using the given data."""
        history = self.model.fit_generator(
            generator=train, validation_data=validation,
            epochs=epochs, shuffle=True, class_weight=class_weight)
        self.training_history.append(
            ({"epochs": epochs, "class_weight": class_weight}, history)
        )
        self.data_ids = {
            "train": train.dataset.labels,
            "validation": validation.dataset.labels if validation else [],
        }
        return history

    def predict(self, data):
        """Predict on a list of cases or som samples."""
        xdata, _ = self.array_from_cases(data)
        preds = self.model.predict(xdata)
        label_preds = [dict(zip(self.binarizer.classes_, pred)) for pred in preds]
        return label_preds

    def predict_generator(self, data: som_dataset.SOMSequence):
        """Predict the given SOM dataset."""
        preds = []
        for pred in self.model.predict_generator(data):
            preds.append(pred)
        pred_arr = np.array(preds)
        pred_labels = self.binarizer.inverse_transform(pred_arr)
        return pred_arr, pred_labels
