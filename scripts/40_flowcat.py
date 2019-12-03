#!/usr/bin/env python3
# pylint: skip-file
# flake8: noqa
import flowcat
import sklearn as sk

cases = flowcat.CaseCollection.load(
    flowcat.utils.URLPath("output/4-flowsom-cmp/samples"),
    flowcat.utils.URLPath("output/4-flowsom-cmp/samples/metadata"))
output_path = flowcat.utils.URLPath("output/4-flowsom-cmp/flowcat-denovo")

model = flowcat.som.CaseSom(
    tubes=cases.selected_markers,
    modelargs={
        "max_epochs": 10,
        "batch_size": 50000,
        "initial_learning_rate": 0.05,  # default: 0.05
        "end_learning_rate": 0.01,  # default: 0.01
        "learning_cooling": "linear",
        "initial_radius": 24,
        "end_radius": 2,
        "radius_cooling": "linear",
        "dims": (32, 32, -1),
    }
)

for case in cases:
    som = model.transform(case, scaler=sk.preprocessing.StandardScaler)
    flowcat.som.base.save_som(som, path=output_path / case.id, subdirectory=False)
