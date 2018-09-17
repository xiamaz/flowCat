"""
Visualization of misclassifications
---
1. Load pretrained models from keras and list of labels of cases to visualize.
2. Create data sources for the selected labels
3. Modify model for visualization.
4. Generate visualizations for data.
"""
import pandas as pd

from sklearn import preprocessing
import keras

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from clustering import collection as cc
from map_class import inverse_binarize


def split_correctness(prediction):
    values = prediction.drop("correct", axis=1)
    truth = prediction["correct"]
    groups = [c for c in prediction.columns if c != "correct"]

    preds = inverse_binarize(values, groups)

    correct = truth == preds
    incorrect_data = prediction.loc[~correct, :].copy()
    correct_data = prediction.loc[correct, :].copy()
    return correct_data, incorrect_data


def sel_high(data):
    sortdata = data.sort_values(data.name, ascending=False)
    seldata = sortdata.iloc[0:5, :]
    return seldata


def get_high_classified(prediction):
    high = prediction.groupby("correct").apply(sel_high)
    return high


def add_infiltration(data, cases):
    labels = list(data.index)
    found = []
    for case in cases:
        if case.id in labels:
            found.append(case)
    found.sort(key=lambda c:labels.index(c.id))
    infiltrations = [c.infiltration for c in found]
    data["infiltration"] = infiltrations
    return data


def plot_grouped_density(data, groupcol):
    ax = plt.gca()
    data = data.loc[data["infiltration"] > 0, :]
    for name, gdata in data.groupby(groupcol):
        sns.distplot(gdata["infiltration"], hist=False, ax=ax, label=name)

    plt.savefig("testdensity.png")


def main():
    cases = cc.CaseCollection("/home/zhao/tmp/CLL-9F", tubes=[1, 2])
    # model = keras.models.load_model("mll-sommaps/models/convolutional/model_0.h5")

    predictions = pd.read_csv("mll-sommaps/models/convolutional/predictions_0.csv", index_col=0)

    correct, wrong = split_correctness(predictions)

    high_correct = get_high_classified(correct)

    wrong_infil = add_infiltration(wrong, cases)

    plot_grouped_density(wrong_infil, "correct")


if __name__ == "__main__":
    main()
