"""
Convenience wrapper for loading a trained model. This will load the model,
associated configuration for preprocessing and the correct binarizer for
transforming group labels to one-hot-encoding.
"""
import keras
from flowcat import utils, som_dataset, io_functions


class SOMClassifier:
    """Wrapper class for keras-based CNN classifier.
    """

    def __init__(self, model, binarizer, config, data_ids: dict = None):
        self.model = model
        self.config = config
        self.binarizer = binarizer
        self.data_ids = data_ids

    @classmethod
    def load(cls, path: utils.URLPath):
        """Load classifier model from the given path."""
        config = io_functions.load_json(path / "config.json")
        model = keras.models.load_model(str(path / "model.h5"))
        binarizer = io_functions.load_joblib(path / "binarizer.joblib")

        data_ids = {
            "validation": io_functions.load_json(path / "ids_validate.json"),
            "train": io_functions.load_json(path / "ids_train.json"),
        }
        return cls(model, binarizer, config, data_ids=data_ids)

    def get_validation_data(self, dataset: som_dataset.SOMDataset) -> som_dataset.SOMDataset:
        return dataset.filter(labels=self.data_ids["validation"])

    def create_sequence(
        self,
        dataset: som_dataset.SOMDataset,
        batch_size: int = 128
    ) -> som_dataset.SOMSequence:

        def getter(data, tube):
            return data.get_tube(tube, kind="som").get_data().data

        seq = som_dataset.SOMSequence(
            dataset, self.binarizer,
            get_array_fun=getter,
            tube=self.config["tubes"],
            batch_size=batch_size,
            pad_width=self.config["pad_width"],
        )
        return seq
