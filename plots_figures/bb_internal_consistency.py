# pylint: skip-file
# flake8: noqa
import sys

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


cases = CaseCollection.from_path("/data/flowcat-data/mll-flowdata/fixedCLL-9F")

# normals = cases.filter(groups=["normal"])
# over_80 = cases.filter(
#     groups=["CLL", "MBL", "MCL", "PL", "LPL", "MZL", "FL", "HCL"],
#     infiltration=80)

cll = cases.filter(groups=["CLL"])
mzl = cases.filter(groups=["MZL"])

clls = sorted(cll.data, key=lambda c: c.infiltration, reverse=True)
mzls = sorted(mzl.data, key=lambda c: c.infiltration, reverse=True)

hi_cll = clls[:100]
hi_mzl = mzls[:100]

# hcl_1 = cases.get_label("ba1de0efbd9ec6847738ac3060b8a3e23b19b6ad")
# hcl_2 = cases.get_label("1b5f9e755fda2d8da248565a5534bcff0d846c93")

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
        "max_epochs": 10,
        "subsample_size": 32,
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

tbpath = "tensorboard/hcltest"
# result = som.create_som([hcl_1], config, tensorboard_path=tbpath)

som_cll = [som.create_som([c], config) for c in hi_cll]
som_mzl = [som.create_som([c], config) for c in hi_mzl]

tsne = TSNE()

tlabels = np.array(["CLL"] * len(som_cll) + ["MZL"] * len(som_mzl))
colors = ["blue", "red"]
for tube in [1, 2, 3]:
    tdata = [d[tube].values.flatten() for d in som_cll + som_mzl]
    transformed = tsne.fit_transform(tdata)
    fig, ax = plt.subplots()
    for i, group in enumerate(["MZL", "CLL"]):
        ax.scatter(
            transformed[tlabels == group, 0],
            transformed[tlabels == group, 1],
            label=group, color=colors[i])
    ax.legend()
    fig.savefig(f"cll_mzl_t{tube}.png")
