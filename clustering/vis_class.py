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

from clustering import collection as cc
from clustering import plotting as cp
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


def plot_tube(case, tube):
    tubecase = case.get_tube(tube)
    fcsdata = tubecase.data

    tubegates = cp.ALL_GATINGS[tube]

    fig = Figure()


    FigureCanvas(fig)
    fig.savefig("test.png")


def main():
    cases = cc.CaseCollection(MLLDATA / "mll-flowdata/CLL-9F", tubes=[1, 2])
    # model = keras.models.load_model("mll-sommaps/models/convolutional/model_0.h5")

    predictions = pd.read_csv(
        MLLDATA / "mll-sommaps/models/smallernet_double_yesglobal_epochrand_mergemult_sommap_8class/predictions_0.csv",
        index_col=0)

    # merged_predictions = merge_predictions(predictions, map_class.GROUP_MAPS["6class"])
    result = map_class.create_metrics_from_pred(predictions, map_class.GROUP_MAPS["6class"]["map"])
    print(result)
    return

    predictions = add_correct_magnitude(predictions)
    predictions = add_infiltration(predictions, cases)

    correct, wrong = split_correctness(predictions)

    false_negative = wrong.loc[wrong["pred"] == "normal", :]
    # print(false_negative.loc[:, ["infiltration", "correct", "largest"]])

    high_infil_false = false_negative.loc[false_negative["infiltration"] > 5.0, :]
    sel_high = high_infil_false.iloc[1, :]
    case = cases.get_label(sel_high.name)
    plot_tube(case, 1)


if __name__ == "__main__":
    main()
