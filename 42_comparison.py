#!/usr/bin/env python3
# pylint: skip-file
# flake8: noqa
from math import ceil

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from flowcat import io_functions, utils, mappings


def plot_ax_scatter(datasets, styles, chan_x, chan_y, ax):
    """Create scatterplots on the given axes object.
    """
    scts = []
    for dataset, style in zip(datasets, styles):
        y = dataset[chan_y]
        x = dataset[chan_x]
        sct = ax.scatter(x, y, **style)
        scts.append(sct)

    ax.set_xlabel(chan_x)
    ax.set_ylabel(chan_y)

    return ax, scts


def plot_som(datasets, styles, tube, path):
    """
    Create a SOM plot from the list of datasets and plot styles.
    """
    chan_y = "SS INT LIN"
    chan_x = "CD45-KrOr"

    views = mappings.PLOT_2D_VIEWS[tube]

    n_cols = 4
    n_rows = ceil(len(views) / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=100)

    for i, ax in enumerate(axes.flatten()):
        try:
            chan_x, chan_y = views[i]
            _, scatters = plot_ax_scatter(datasets, styles, chan_x, chan_y, ax)
        except IndexError:
            ax.axis("off")

    labels = tuple(style.get("label", "") for style in styles)
    fig.legend(scatters, labels, loc="lower right", fontsize="large", markerscale=4)
    fig.tight_layout()
    path = path.with_suffix(".png")
    with path.open("wb") as figfile:
        plt.savefig(figfile)
    plt.close("all")


def load_flowsom_data(case_id, flowsom_path, tubes):
    """Load a given flowsom data sample into a dict of pandas dataframes."""
    soms = {}
    for tube in tubes:
        sompath = flowsom_path / f"{case_id}_t{tube}.csv"
        soms[tube] = io_functions.load_csv(sompath)
    return soms


def load_flowcat_data(case_id, flowcat_path, tubes):
    """Load given flowcat data into a dict of pandas dataframes."""
    soms = {}
    config = io_functions.load_json(flowcat_path + "_config.json")
    for tube in tubes:
        sompath = flowcat_path / f"{case_id}_t{tube}.npy"
        channels = config[tube]["channels"]
        soms[tube] = pd.DataFrame(np.load(sompath).reshape((-1, len(channels))), columns=channels)
    return soms


def load_fcs_data(case, tubes):
    fcs = {}
    for tube in tubes:
        fcs_data = case.get_tube(tube, kind="fcs").get_data()
        fcs_df = fcs_data.data
        fcs_meta = fcs_data.meta
        columns = fcs_data.channels

        # for i, channel in enumerate(columns):
        #     ch_meta = fcs_meta[channel]
        #     if ch_meta.pne == (0.0, 0.0) and ch_meta.png != 1.0:
        #         fcs_array[i] *= ch_meta.png

        fcs_df = StandardScaler().fit_transform(fcs_df)
        fcs[tube] = pd.DataFrame(fcs_df, columns=columns)
    return fcs


def get_case_data(case, tubes, paths):
    data = {}
    for name, path in paths.items():
        if name == "fcs":
            data[name] = load_fcs_data(case, tubes)
        elif name == "flowsom":
            data[name] = load_flowsom_data(case.id, path, tubes)
        elif name == "flowcat":
            data[name] = load_flowcat_data(case.id, path, tubes)
        else:
            data[name] = load_flowsom_data(case.id, path, tubes)
    return data


def plot_hexbin_cmp_all(datas, tubes, output):
    for tube in tubes:
        views = mappings.PLOT_2D_VIEWS[tube]
        for chan_x, chan_y in views:
            fig, axes = plt.subplots(1, len(datas), figsize=(6 * len(datas), 5), dpi=100)
            for ax, (name, data) in zip(axes, datas.items()):
                ax.hexbin(data[tube][chan_x], data[tube][chan_y], cmap="Blues", gridsize=32)
                ax.set_xlabel(chan_x)
                ax.set_ylabel(chan_y)
                ax.set_title(name)
            plt.savefig(str(output / f"t{tube}_{chan_x}_{chan_y}.png"))
            plt.close("all")


def plot_single_case(case, tubes, output, flowsom_path, flowcat_path):
    flowsom_data = load_flowsom_data(case.id, flowsom_path, tubes)
    flowcat_data = load_flowcat_data(case.id, flowcat_path, tubes)
    fcs_data = load_fcs_data(case, tubes)
    for tube in ("1", "2", "3"):
        name = f"{case.id}_t{tube}"
        plot_som(
            (
                fcs_data[tube],
                flowsom_data[tube],
                flowcat_data[tube],
            ),
            (
                {"s": 1, "marker": ".", "color": "grey", "label": "fcs"},
                {"s": 8, "marker": ".", "color": "blue", "label": "flowCat", "alpha": 0.5},
                {"s": 8, "marker": ".", "color": "red", "label": "flowSOM", "alpha": 0.5},
            ),
            tube,
            output / name)


cases = io_functions.load_case_collection(
    utils.URLPath("output/4-flowsom-cmp/samples"),
    utils.URLPath("output/4-flowsom-cmp/samples/samples.json"))

# Compare flowsom results with flowcat results. Do keep in mind that they are
# scaled differently

# flowsom_path = utils.URLPath("output/4-flowsom-cmp/flowsom-samples")
# flowcat_path = utils.URLPath("output/4-flowsom-cmp/flowcat-denovo")
# flowcat_ref_path = flowcat.utils.URLPath("output/4-flowsom-cmp/flowcat-refsom")

output = utils.URLPath("output/4-flowsom-cmp/figures-refit")
tubes = ("1", "2", "3")
groups = set(cases.groups)

from collections import defaultdict


def merge_case_data(case_datas):
    data_lists = defaultdict(lambda: defaultdict(list))
    for case_data in case_datas:
        for kind, tube_data in case_data.items():
            for tube, data in tube_data.items():
                data_lists[kind][tube].append(data)

    merged_data = defaultdict(dict)
    for kind, tube_data in data_lists.items():
        for tube, data_list in tube_data.items():
            merged_data[kind][tube] = pd.concat(data_list)
    return merged_data


flowsom_sample_path = utils.URLPath("output/4-flowsom-cmp/flowsom-10")
flowcat_sample_path = utils.URLPath("output/4-flowsom-cmp/flowcat-refit-s10")
paths = {
    "fcs": None,
    "flowsom": flowsom_sample_path,
    "flowcat": flowcat_sample_path,
}

groups = ["HCL"]

for group in groups:
    case_datas = []
    for case in cases.filter(groups=[group]):
        case_data = get_case_data(case, tubes, paths)
        case_datas.append(case_data)
    merged_data = merge_case_data(case_datas)
    plotpath = output / "merged" / group
    plotpath.mkdir()
    plot_hexbin_cmp_all(merged_data, tubes=tubes, output=plotpath)


for case in cases:
    plot_single_case(case, tubes, output / "size10", flowsom_sample_path, flowcat_sample_path)

flowsom_sample_path = utils.URLPath("output/4-flowsom-cmp/flowsom-32")
flowcat_sample_path = utils.URLPath("output/4-flowsom-cmp/flowcat-refsom")
paths = {
    "fcs": None,
    "flowsom": flowsom_sample_path,
    "flowcat retrained": flowcat_sample_path,
    "flowcat random": utils.URLPath("output/4-flowsom-cmp/flowcat-denovo"),
}

for group in groups:
    case_datas = []
    for case in cases.filter(groups=[group]):
        case_data = get_case_data(case, tubes, paths)
        case_datas.append(case_data)
    merged_data = merge_case_data(case_datas)
    plotpath = output / "mergedmore32" / group
    plotpath.mkdir()
    plot_hexbin_cmp_all(merged_data, tubes=tubes, output=plotpath)

for case in cases:
    plot_single_case(case, tubes, output / "size32", flowsom_sample_path, flowcat_sample_path)
