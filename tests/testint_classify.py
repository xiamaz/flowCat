"""
Test training of classifier, as well as saving and loading to use
other classifiers.
"""
import unittest

from flowcat import classify, configuration
from flowcat.dataset import combined_dataset, case

from . import shared


def create_basic_configuration():
    return configuration.ClassificationConfig({
        "name": "basictest",
        "dataset": {
            "names": {
                "FCS": str(shared.FCS_PATH),
                "SOM": str(shared.DATAPATH / "indivsom"),
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


def create_single_case(name):
    filepaths = [
        {"fcs": {"path": f"{name}_{i}.lmd"}, "date": "2018-01-02", "tube": i} for i in [1, 2]
    ]
    tdict = {
        "id": name,
        "date": "2018-01-02",
        "filepaths": filepaths,
    }
    return case.Case(tdict, path=shared.DATAPATH / "fcs")


class TestClassify(unittest.TestCase):
    """Testing of SOM classifier."""

    def test_basic_train(self):
        """Training a basic model from an existing SOM dataset."""
        config = create_basic_configuration()

        dataset = combined_dataset.CombinedDataset.from_config(config)
        train, test = combined_dataset.split_dataset(dataset, **config("split"), seed=42)

        model, trainseq, testseq = classify.generate_model_inputs(train, test, config("model"))
        classify.fit(model, trainseq)
        classify.predict_generator(model, testseq)

        classify.save_model(model, trainseq, config, shared.DATAPATH / "basicmodel", dataset=dataset)

    def test_load_model(self):
        """Loading a model and predicting another case from FCS files."""
        single = create_single_case("cll1")
        model, transformer, groups = classify.load_model(shared.DATAPATH / "basicmodel")

        indata = transformer([single])
        pred_df = classify.predict(model, indata, groups, [single.id])
        print(pred_df)
