#!/usr/bin/env python3
# pylint: skip-file
# flake8: noqa
from math import ceil

import flowcat
import pandas as pd
import sklearn as sk
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


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

    views = flowcat.mappings.PLOT_2D_VIEWS[tube]

    n_cols = 4
    n_rows = ceil(len(views) / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=100)

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


cases = flowcat.CaseCollection.load(
    flowcat.utils.URLPath("output/4-flowsom-cmp/samples"),
    flowcat.utils.URLPath("output/4-flowsom-cmp/samples/metadata"))

# Compare flowsom results with flowcat results. Do keep in mind that they are
# scaled differently
flowsom_path = flowcat.utils.URLPath("output/4-flowsom-cmp/flowsom-samples")
flowcat_path = flowcat.utils.URLPath("output/4-flowsom-cmp/flowcat-denovo")
flowcat_ref_path = flowcat.utils.URLPath("output/4-flowsom-cmp/flowcat-refsom")

comparison_output = flowcat.utils.URLPath("output/4-flowsom-cmp/comparison-denovo")

flowcat_path = flowcat_path

for case in cases:
    sompath = flowsom_path / case.id
    som = flowcat.load_som(sompath, tube=("1", "2", "3"))
    catpath = flowcat_path / case.id
    catsom = flowcat.load_som(catpath, tube=("1", "2", "3"))
    for tube in ("1", "2", "3"):
        sample = case.get_tube(tube)
        norm = sk.preprocessing.StandardScaler().fit_transform(sample.data.data)
        norm = pd.DataFrame(norm, columns=sample.data.data.columns)
        somtube = som.get_tube(tube)
        cattube = catsom.get_tube(tube)
        name = f"{case.id}_t{tube}"
        plot_som(
            (
                norm,
                cattube.data,
                somtube.data,
            ),
            (
                {"s": 1, "marker": ".", "color": "grey", "label": "fcs"},
                {"s": 8, "marker": ".", "color": "blue", "label": "flowCat", "alpha": 0.5},
                {"s": 8, "marker": ".", "color": "red", "label": "flowSOM", "alpha": 0.5},
            ),
            tube,
            comparison_output / name)
