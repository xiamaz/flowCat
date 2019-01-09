"""
TSNE separation of our initial FCS data without additional processing.
"""
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


OUTPUT_DIR = "output/fcs-tsne"


def create_infil_sorted_group_list(cases, group):
    filtered = cases.filter(groups=[group], tubes=[1, 2, 3])
    # remove zero infiltration from cohorts not normal
    if group != "normal":
        filtered = [c for c in filtered if c.infiltration > 0]
    return sorted(filtered, key=lambda c: c.infiltration, reverse=True)


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
    group + "_hi": create_infil_sorted_group_list(cases, group)[:num]
    for group, num in groups.items()
}

lo_groups = {
    group + "_lo": create_infil_sorted_group_list(cases, group)[::-1][:num]
    for group, num in groups.items()
}

def get_fcs(case, tube):
    tcase = case.get_tube(tube)
    return tcase.data.scale().data.sample(1000).values.flatten()

sel_groups = dict(hi_groups, **lo_groups)
colors = ["blue", "red", "green", "orange", "gray", "darkblue", "darkred", "darkgreen", "darkorange", "black"]

tlabels = np.array([group for group, cases in sel_groups.items() for _ in cases])

for tube in [1, 2, 3]:
    tdata = [get_fcs(case, tube) for cases in sel_groups.values() for case in cases]

    tsne = TSNE(random_state=SEED)
    transformed = tsne.fit_transform(tdata)
    fig, ax = plt.subplots()
    for i, group in enumerate(sel_groups):
        print(colors[i])
        ax.scatter(
            transformed[tlabels == group, 0],
            transformed[tlabels == group, 1],
            label=group, color=colors[i])
    ax.legend()
    ax.set_title(f"FCS: Scaled Sample 1000 Tube {tube}")
    fig.savefig(f"fcs_scaled_s1000_t{tube}.png")

# tbpath = "tensorboard/hcltest"
# result = som.create_som([hcl_1], config, tensorboard_path=tbpath)
