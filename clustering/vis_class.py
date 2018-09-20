"""
Visualization of misclassifications
---
1. Load pretrained models from keras and list of labels of cases to visualize.
2. Create data sources for the selected labels
3. Modify model for visualization.
4. Generate visualizations for data.
"""
import os
import pathlib
import collections

import numpy as np
import pandas as pd

from sklearn import preprocessing
import keras

# import matplotlib as mpl
# mpl.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
import seaborn as sns

from color2D import Color2D

from clustering import collection as cc
from clustering import plotting as cp
from clustering.transformation import pregating, tfsom
from map_class import inverse_binarize
import map_class


if "MLLDATA" in os.environ:
    MLLDATA = pathlib.Path(os.environ["MLLDATA"])
else:
    MLLDATA = pathlib.Path()


def split_correctness(prediction):
    correct = prediction["correct"] == prediction["pred"]
    incorrect_data = prediction.loc[~correct, :].copy()
    correct_data = prediction.loc[correct, :].copy()
    return correct_data, incorrect_data


def add_infiltration(data, cases):
    labels = list(data.index)
    found = []
    for case in cases:
        if case.id in labels:
            found.append(case)
    found.sort(key=lambda c: labels.index(c.id))
    infiltrations = [c.infiltration for c in found]
    data["infiltration"] = infiltrations
    return data


def add_correct_magnitude(data):
    newdata = data.copy()
    valcols = [c for c in data.columns if c != "correct"]
    selval = np.vectorize(lambda i: valcols[i])
    newdata["largest"] = data[valcols].max(axis=1)
    newdata["pred"] = selval(data[valcols].values.argmax(axis=1))
    return newdata

# def plot_grouped_density(data, groupcol):
#     ax = plt.gca()
#     data = data.loc[data["infiltration"] > 0, :]
#     for name, gdata in data.groupby(groupcol):
#         sns.distplot(gdata["infiltration"], hist=False, ax=ax, label=name)
#
#     plt.savefig("testdensity.png")


def pregating_selection(fcsdata):
    gater = pregating.SOMGatingFilter()
    transdata = gater.fit_transform(fcsdata)
    nontransdata = fcsdata.drop(transdata.index)
    pregating_selection = [
        (nontransdata.index, "grey", "other"),
        (transdata.index, "blue", "lymphocytes"),
    ]
    return pregating_selection


def get_sommap_tube(dataset_path, label, tube):
    path = dataset_path / f"{label}_t{tube}.csv"
    data = pd.read_csv(path, index_col=0)
    return data


def map_fcs_to_sommap(case, tube, sommap_path):
    """Map for the given case the fcs data to their respective sommap data."""
    sommap_data = get_sommap_tube(sommap_path, case.id, tube)
    counts = sommap_data["counts"]
    sommap_data.drop(["counts", "count_prev"], inplace=True, errors="ignore", axis=1)
    gridwidth = int(np.round(np.sqrt(sommap_data.shape[0])))

    tubecase = case.get_tube(1)
    # get scaled zscores
    fcsdata = tubecase.get_data()

    model = tfsom.TFSom(
        m=gridwidth, n=gridwidth, channels=sommap_data.columns,
        initialization_method="reference", reference=sommap_data,
        max_epochs=0)

    model.train([fcsdata], num_inputs=1)
    mapping = model.map_to_nodes(fcsdata)

    fcsdata = tubecase.data

    fcsdata["somnode"] = mapping
    return fcsdata, gridwidth


def nodenum_to_coord(nodenum, gridwidth=32, relative=True):
    x = nodenum % gridwidth
    y = int(nodenum / gridwidth)

    return x / gridwidth, y / gridwidth


def sommap_selection(fcsdata, gridwidth=32):
    palette = Color2D()
    selection = []
    for name, gdata in fcsdata.groupby("somnode"):
        coords = nodenum_to_coord(name, gridwidth=gridwidth, relative=True)
        color = palette[coords] / 255
        selection.append((gdata.index, color, name))

    return selection


def plot_tube(case, tube, title="Scatterplots", selection=None, sommappath=""):
    tubecase = case.get_tube(tube)
    fcsdata = tubecase.data

    tubegates = cp.ALL_GATINGS[tube]

    nodedata = map_fcs_to_sommap(case, tube, sommappath)

    if selection == "pregating":
        chosen_selection = pregating_selection(tubecase.data)
    elif selection == "sommap":
        fcsmapped, gridwidth = map_fcs_to_sommap(case, tube, sommappath)
        chosen_selection = sommap_selection(fcsmapped, gridwidth=gridwidth)
    else:
        chosen_selection = None

    fig = Figure(figsize=(12, 8), dpi=300)
    for i, gating in enumerate(tubegates):
        axes = fig.add_subplot(3, 4, i + 1)

        cp.scatterplot(fcsdata, gating, axes, selections=chosen_selection)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(f"{title} Tube {tube}")
    FigureCanvas(fig)
    fig.savefig("test.png")


def main():
    sommap_dataset = MLLDATA / "mll-sommaps/sample_maps/selected1_toroid_s32"

    cases = cc.CaseCollection(MLLDATA / "mll-flowdata/CLL-9F", tubes=[1, 2])
    # model = keras.models.load_model("mll-sommaps/models/convolutional/model_0.h5")

    predictions = pd.read_csv(
        MLLDATA / "mll-sommaps/models/smallernet_double_yesglobal_epochrand_mergemult_sommap_8class/predictions_0.csv",
        index_col=0)

    # merged_predictions = merge_predictions(predictions, map_class.GROUP_MAPS["6class"])
    # result = map_class.create_metrics_from_pred(predictions, map_class.GROUP_MAPS["6class"]["map"])
    # print(result)

    predictions = add_correct_magnitude(predictions)
    predictions = add_infiltration(predictions, cases)

    correct, wrong = split_correctness(predictions)

    false_negative = wrong.loc[wrong["pred"] == "normal", :]
    # print(false_negative.loc[:, ["infiltration", "correct", "largest"]])

    high_infil_false = false_negative.loc[false_negative["infiltration"] > 5.0, :]
    sel_high = high_infil_false.iloc[1, :]
    case = cases.get_label(sel_high.name)
    plot_tube(case, 1, sommappath=sommap_dataset, selection="sommap")


if __name__ == "__main__":
    main()
