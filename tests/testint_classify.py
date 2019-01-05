"""
Test training of classifier, as well as saving and loading to use
other classifiers.
"""
import unittest

from flowcat import classify, configuration
from flowcat.dataset import combined_dataset

from . import shared


def create_basic_configuration():
    return configuration.ClassificationConfig({
        "name": "basictest",
        "dataset": {
            "names": {
                "FCS": str(shared.FCS_PATH),
                "SOM": str(shared.SOM_PATH),
            },
            "filters": {
                "num": 20,
            },
        },
        "model": {
            "type": "som",
            "train_args": {
                "batch_size": 32,
            },
            "test_args": {
                "batch_size": 64,
            }
        },
        "fit": {
            "train_epochs": 2,
        },
    })


class TestClassify(unittest.TestCase):
    """Testing of SOM classifier."""

    def test_basic_train(self):
        """Training a basic model from an existing SOM dataset."""
        config = create_basic_configuration()

        dataset = combined_dataset.CombinedDataset.from_config(config)
        train, test = combined_dataset.split_dataset(dataset, **config("split"), seed=42)

        model, trainseq, testseq = classify.generate_model_inputs(train, test, config("model"))
        classify.fit(model, trainseq)
        classify.predict(model, testseq)

        classify.save_model(model, config, shared.DATAPATH / "lolmodel", dataset=dataset)

    def test_load_model(self):
        """Loading a model and predicting another case from FCS files."""
        pass
