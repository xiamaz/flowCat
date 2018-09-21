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
import vis

# import matplotlib as mpl
# mpl.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
# import matplotlib.pyplot as plt
import seaborn as sns

#from color2D import Color2D
import colorsys

from clustering import collection as cc
from clustering import plotting as cp
from clustering.transformation import pregating, tfsom
from map_class import inverse_binarize
import map_class
import saliency
import consensus_cases


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
    sommap_data.drop(["counts", "count_prev"],
                     inplace=True, errors="ignore", axis=1)
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


def sommap_selection(fcsdata, grads, gridwidth=32):
    #palette = Color2D()
    selection = []
    grad_colors = cm.ScalarMappable(cmap='jet').to_rgba(grads)
    for name, gdata in fcsdata.groupby("somnode"):
        #coords = nodenum_to_coord(name, gridwidth=gridwidth, relative=True)
        #color = cm.jet(round(256*grads[name]))
        color = grad_colors[name]
        color[3] = grads[name]
        selection.append((gdata.index, color, name))
    return selection


def plot_tube(case, tube, grads, title="Scatterplots", selection=None, sommappath=""):
    tubecase = case.get_tube(tube)
    fcsdata = tubecase.data

    tubegates = cp.ALL_GATINGS[tube]

    nodedata = map_fcs_to_sommap(case, tube, sommappath)

    if selection == "pregating":
        chosen_selection = pregating_selection(tubecase.data)
    elif selection == "sommap":
        fcsmapped, gridwidth = map_fcs_to_sommap(case, tube, sommappath)
        chosen_selections = []
        for grad in grads:
            mapped_gradients = [grad[index] for index in fcsmapped['somnode']]
            fcsmapped['gradients'] = pd.Series(mapped_gradients,index = fcsmapped.index)
            chosen_selection = sommap_selection(fcsmapped.sort_values(by='gradients',ascending = False), grad, gridwidth=gridwidth)
            fcsmapped.drop('gradients',1,inplace=True)
            chosen_selections.append(chosen_selection)
    else:
        chosen_selection = None

    for idx,selection in enumerate(chosen_selections):
        fig = Figure(figsize=(12, 8), dpi=300)
        for i, gating in enumerate(tubegates):
            axes = fig.add_subplot(3, 4, i + 1)
            cp.scatterplot( fcsdata, gating, axes, selections=selection)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.suptitle(f"{title} Tube {tube} Class {idx}")
        FigureCanvas(fig)
        fig.savefig(f"{title} Tube {tube} Class {idx}.png")


def calc_saliency(case, model, indata, labels, config, preds, classes=None):
    # visualize the last layer
    layer_idx = -1

    # load existing model
    model = keras.models.load_model(model)

    # modify model for saliency usage
    model.layers[layer_idx].activation = keras.activations.linear
    model = vis.utils.utils.apply_modifications(model)

    # load sommap dataset
    dataset = saliency.dataset_from_config(
        indata, labels, config, batch_size=1)
    xdata, ydata = dataset.get_batch_by_label(case.id)

    input_indices = [*range(len(xdata))]
    corr_group = preds.loc[case.id, "correct"]
    pred_group = preds.loc[case.id, "pred"]

    if classes == None:
        classes = [corr_group,pred_group]

    gradients = [vis.visualization.visualize_saliency(model, layer_idx, dataset.groups.index(group), seed_input=xdata, input_indices=input_indices) for group in set(classes)]

    #regroup gradients into tube1 and tube2
    gradients = [[grad[0].flatten() for grad in gradients],[grad[1].flatten() for grad in gradients]]

    return gradients


def main():
    c_indata = MLLDATA / "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/data_paths.p"
    c_model = MLLDATA / "mll-sommaps/models/smallernet_double_yesglobal_epochrand_sommap_8class/model_0.h5"
    c_config = MLLDATA / "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/config.json"
    c_labels = MLLDATA / "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/test_labels.json"
    c_preds = MLLDATA / "mll-sommaps/models/smallernet_double_yesglobal_epochrand_sommap_8class/predictions_0.csv"
    c_tube = 1
    sommap_dataset = MLLDATA / "mll-sommaps/sample_maps/selected1_toroid_s32"

    cases = cc.CaseCollection(MLLDATA / "mll-flowdata/CLL-9F", tubes=[1, 2])
    # model = keras.models.load_model("mll-sommaps/models/convolutional/model_0.h5")

    predictions = pd.read_csv(
        MLLDATA / "mll-sommaps/models/smallernet_double_yesglobal_epochrand_mergemult_sommap_8class/predictions_0.csv",
        index_col=0)

    #merged_predictions = merge_predictions(predictions, map_class.GROUP_MAPS["6class"])
    #result = map_class.create_metrics_from_pred(predictions, map_class.GROUP_MAPS["6class"]["map"])
    #print(result)

    predictions = add_correct_magnitude(predictions)
    predictions = add_infiltration(predictions, cases)

    correct, wrong = split_correctness(predictions)

    false_negative = wrong.loc[wrong["pred"] == "normal", :]
    # print(false_negative.loc[:, ["infiltration", "correct", "largest"]])

    high_infil_false = false_negative.loc[false_negative["infiltration"] > 5.0, :]
    sel_high = high_infil_false.iloc[1, :]
    case = cases.get_label(sel_high.name)
    gradients = calc_saliency(case,c_model, c_indata, c_labels, c_config, predictions)
    plot_tube(case, c_tube, gradients[c_tube-1],
              sommappath=sommap_dataset, selection="sommap")


if __name__ == "__main__":
    main()
