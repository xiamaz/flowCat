#!/usr/bin/env python3
# flake8: noqa
# pylint: skip-file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import bspline

from report import prediction, file_utils
from report.stats import accuracy


def experiment_accuracies(sel_exps):
    """Get experiment accuracies."""
    res = []
    for _, sel_row in sel_exps.iterrows():
        pdata = file_utils.load_predictions(sel_row["predictions"])
        sel_accs = accuracy.acc_table_dfs(pdata)
        # average values over all sets
        res.append(sel_accs)

    avg_accs = pd.concat(res)
    return avg_accs.mean(level=1)


# get different accuracies for the selected set of experiments
pred = prediction.Prediction()
sel_normal = pred.classification.loc["abstract_single_groups_sqrt", "normal", "random"]
sel_somgated = pred.classification.loc["abstract_single_groups_sqrt", "somgated", "random"]

print(experiment_accuracies(sel_normal))
print(experiment_accuracies(sel_somgated))



def add_infiltration(data, infiltrations):
    infil = infiltrations[data.name]
    data["infiltration"] = infil
    return data

def sum_rel_count(data):
    label, group = data.name
    rel_count = data["rel_count"].sum()
    return pd.Series(data={"group": group, "rel_count": rel_count})


mis = Misclassifications(path_classification="../output/classification")

# select infiltration dataset and get misclassifications and infiltrations
infil_set = mis.data.loc["cd5_threeclass", "somcombined_dedup"]
misclass = mis.get_misclassifications(infil_set)
labels = misclass.index.get_level_values("id")

# add infiltration information
test = misclass.groupby("id").apply(
    lambda x: add_infiltration(x, mis.get_infiltrations(infil_set))
)

merged_misclass = misclass.groupby(level=["id", "group"]).apply(
    sum_rel_count
).reset_index(level="group", drop=True)
all_groups = mis.get_groups(infil_set)
all_groups.index.rename("id", inplace=True)
all_infiltrations = mis.get_infiltrations(infil_set)
all_infiltrations.index.rename("id", inplace=True)

misclass_infil = merged_misclass.join(all_infiltrations, how="outer")
misclass_infil["rel_count"].fillna(0.0, inplace=True)
misclass_infil["rel_correct"] = 1 - misclass_infil["rel_count"]
misclass_infil["group"] = all_groups  # add all groups

bins = pd.IntervalIndex.from_tuples([
    (i, i+1) for i in range(0, 100, 10)
])
indices = [d.mid for d in bins]

misclass_infil_over0 = misclass_infil.loc[misclass_infil["infiltration"] > 0]
misclass_infil_over0["bins"] = pd.cut(
    misclass_infil_over0["infiltration"], bins=bins
)

colors = {
    "corr": [
        (0, 1, 0),
        (0, 0.6, 0.4),
        (0, 0.8, 1)
    ],
    "missed": [
        (1, 0, 0),
        (0.6, 0, 0.4),
        (0.8, 0, 1),
    ]
}

def get_color(i, name, alpha=1):
    return (*colors[name][i], alpha)

xnew = np.linspace(0, 100, 1000)
hlist = [r"\\", r"/", r"|", r"-"]
lsstyles = ["-", "--", ":", "-."]

plt.figure()
for i, (group, sel_data) in enumerate(misclass_infil_over0.groupby("group")):
    missed_dist = sel_data.groupby("bins").apply(
        lambda d: d["rel_count"].sum()
    )
    correct_dist = sel_data.groupby("bins").apply(
        lambda d: d["rel_correct"].sum()
    )
    missed_smooth = spline(indices, missed_dist.values, xnew)
    correct_smooth = spline(indices, correct_dist.values, xnew)
    x = indices
    y = missed_dist.values
    plt.plot(
        x, y,
        color=get_color(i, "missed", 1),
        label="Missed {}".format(group),
        ls=lsstyles[i]
    )
    plt.fill_between(x, y, color=get_color(i, "missed", 0.4), hatch=hlist[i])
    y = correct_dist.values
    plt.plot(
        x, y,
        color=get_color(i, "corr", 1),
        label="Correct {}".format(group),
        ls=lsstyles[i]
    )
    plt.fill_between(x, y, color=get_color(i, "corr", 0.4), hatch=hlist[i])

    # missed_dist.plot(
    #     color=missed_colors[i], label="Missed {}".format(group)
    # )
    # correct_dist.plot(
    #     color=corr_colors[i], label="Correct {}".format(group)
    # )
plt.xticks(np.arange(0, 100, step=10))
plt.xlabel("Infiltration")
plt.ylabel("Number of cases")
plt.legend()
plt.savefig("misclassification_infiltration_per_group.png")
plt.close()

# select only ones with infiltration higher than 0%
infiltrations = mis.get_infiltrations(infil_set).reset_index()
infiltrations = infiltrations.loc[infiltrations["infiltration"] > 0]

# get the misclassified and non misclassified infiltrations
misclass_infil = infiltrations.loc[
    infiltrations["label"].isin(labels), "infiltration"
]
non_misclass_infil = infiltrations.loc[
    ~(infiltrations["label"].isin(labels)), "infiltration"
]

plt.figure()
plt.hist(
    misclass_infil,
    bins=range(0, 100, 1),
    color=(1, 0, 0, 0.5),
    label="Wrong",
)
plt.hist(
    non_misclass_infil,
    bins=range(0, 100, 1),
    color=(0, 1, 0, 0.5),
    label="Correct",
)
plt.legend()
plt.savefig("misclassification_histogram.png")
plt.close()

print(
    misclass_infil.mean(),
    misclass_infil.min(),
    misclass_infil.max(),
    misclass_infil.std(),
)

print(
    non_misclass_infil.mean(),
    non_misclass_infil.min(),
    non_misclass_infil.max(),
    non_misclass_infil.std(),
)
print(mean_confidence_interval(misclass_infil))
print(mean_confidence_interval(non_misclass_infil))

hi_infil_misclass = misclass_infil[misclass_infil["infiltration"] > 20.0]
print(hi_infil_misclass)
