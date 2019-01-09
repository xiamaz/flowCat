# pylint: skip-file
# flake8: noqa
import sys
import logging
import pathlib

import numpy as np

from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# adding the parent directory to the search path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from flowcat.dataset.case_dataset import CaseCollection
from flowcat import som, configuration, mappings


def create_infil_sorted_group_list(cases, group):
    filtered = cases.filter(groups=[group], tubes=[1, 2, 3])
    # remove zero infiltration from cohorts not normal
    if group != "normal":
        filtered = [c for c in filtered if c.infiltration > 0]
    return sorted(filtered, key=lambda c: c.infiltration, reverse=True)


def create_subsample_size_tests():

    target_size = 10240

    configs = {}
    for subsample_size in [16, 64, 256, 1024]:
        config = configuration.SOMConfig({
            "name": "hcltest",
            "dataset": {
                "filters": {
                    "tubes": [1, 2, 3],
                },
                "selected_markers": mappings.CHANNEL_CONFIGS["CLL-9F"],
                "preprocessing": "scale",
            },
            "tfsom": {
                "model_name": "hcltest",
                "map_type": "toroid",
                "max_epochs": int(target_size / subsample_size),
                "subsample_size": subsample_size,
                "initial_learning_rate": 0.5,
                "end_learning_rate": 0.1,
                "learning_cooling": "linear",
                "initial_radius": 16,
                "end_radius": 1,
                "radius_cooling": "linear",
                "node_distance": "euclidean",
                "initialization_method": "random",
            },
        })
        configs[f"sample_{subsample_size}"] = config
    return configs


def sample_more_epochs():

    configs = {}
    # for epochs in [10, 50, 100]:
    for epochs in [10, 50, 100]:
        config = configuration.SOMConfig({
            "name": "hcltest",
            "dataset": {
                "filters": {
                    "tubes": [1, 2, 3],
                },
                "selected_markers": mappings.CHANNEL_CONFIGS["CLL-9F"],
            },
            "tfsom": {
                "model_name": "hcltest",
                "map_type": "toroid",
                "max_epochs": epochs,
                "subsample_size": 1024,
                "initial_learning_rate": 0.5,
                "end_learning_rate": 0.1,
                "learning_cooling": "linear",
                "initial_radius": 16,
                "end_radius": 1,
                "radius_cooling": "linear",
                "node_distance": "euclidean",
                "initialization_method": "random",
            },
        })
        configs[f"epoch_{epochs}"] = config
    return configs


SEED = 42


cases = CaseCollection.from_path("/data/flowcat-data/mll-flowdata/fixedCLL-9F")

groups = {
    "CLL": 20,
    "MZL": 20,
    "HCL": 5,
    "FL": 5,
    "normal": 20,
}

hi_groups = {
    group: create_infil_sorted_group_list(cases, group)[:num]
    for group, num in groups.items()
}

lo_groups = {
    group: create_infil_sorted_group_list(cases, group)[::-1][:num]
    for group, num in groups.items()
}


# configs = create_subsample_size_tests()
configs = sample_more_epochs()
sel_groups = hi_groups

for name, config in configs.items():
    print(f"Generating {name}")
    som_groups = {
        case.id: {
            "som": som.create_som([case], config, seed=SEED),
            "group": case.group,
        }
        for group, cases in sel_groups.items()
        for case in cases
    }
    tsne = TSNE(random_state=SEED)

    colors = ["blue", "red", "green", "brown", "black"]
    tlabels = np.array([case["group"] for case in som_groups.values()])
    for tube in [1, 2, 3]:
        tdata = [case["som"][tube].values.flatten() for case in som_groups.values()]

        transformed = tsne.fit_transform(tdata)
        fig, ax = plt.subplots()
        for i, group in enumerate(groups):
            ax.scatter(
                transformed[tlabels == group, 0],
                transformed[tlabels == group, 1],
                label=group, color=colors[i])
        ax.legend()
        ax.set_title(f"Scaled only Hi infiltration {name} Tube {tube}")
        fig.savefig(f"scaled_{name}_t{tube}.png")

# tbpath = "tensorboard/hcltest"
# result = som.create_som([hcl_1], config, tensorboard_path=tbpath)
