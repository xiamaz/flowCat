#!/usr/bin/env python3
# flake8: noqa
# pylint: skip-file
"""Visualization of separation in output data of classification."""
import sys
sys.path.insert(0, "../classification")

from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from contextlib import contextmanager

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

from classification.upsampling import InputData


def parse_arguments():
    """Get paths and desires processing method from command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("input", help="CSV input data.")
    parser.add_argument("method", help="Analysis method.", type=str.lower)
    parser.add_argument("output", help="Output directory for plots.")

    parser.add_argument(
        "--pattern", help="Match the respective pattern in the given folder.",
        default=""
    )

    args = parser.parse_args()
    return args


@contextmanager
def new_plot(path, title=""):
    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)

    yield ax

    ax.set_title(title)
    FigureCanvas(fig)
    fig.savefig(path)


def scatterplot(data, labels, uniques, ax):
    colors = [
        cm.Set1(float(i) / len(uniques))
        for i in range(len(uniques))
    ]

    for i, unique in enumerate(uniques):
        selection = data[labels == i]
        ax.scatter(
            selection[:, 0], selection[:, 1], c=colors[i], label=unique,
            s=1, marker="."
        )
    ax.legend(markerscale=10)
    return ax


def upsampling_histogram_plot(data, path="upsampling_histo", title=""):
    groups = data.groupby("group")
    fig = Figure()
    for i, (group, gdata) in enumerate(groups):
        mean_data = gdata.mean()
        std_data = gdata.std()
        ax = fig.add_subplot(5, 2, i + 1)
        plot_histogram((mean_data, std_data), ax, title=group)

    fig.set_size_inches(20, 10)
    FigureCanvas(fig)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=100)


def create_plot(transfun, title, path, index, data, label, unique):
    tdata = transfun(data)

    plotpath = str(path) + "_{}".format(index)

    with new_plot(plotpath, title) as ax:
        scatterplot(tdata, label, unique, ax)


def create_histogram(title, path, index, data, label, unique):
    combined_data = data.copy()
    combined_data["group"] = label

    plotpath = str(path) + "_{}".format(index)
    upsampling_histogram_plot(combined_data, plotpath, title)


def plot_creator(args):
    (path, method), items = args
    dimensions = 2
    if method == "pca":
        model = PCA(n_components=dimensions)
        title = "PCA in {} dimensions".format(dimensions)
        create_plot(model.fit_transform, title, path, *items)
    elif method == "tsne":
        model = TSNE(n_components=dimensions)
        title = "tSNE in {} dimensions".format(dimensions)
        create_plot(model.fit_transform, title, path, *items)
    elif method == "histo":
        title = "Histogram visualization with standard deviation"
        create_histogram(title, *items)
    else:
        raise RuntimeError("Unknown method {}".format(method))


class Visualizer:
    def __init__(self, inpath, outpath, pattern):
        self._inpath = Path(inpath)
        self._outpath = Path(outpath)
        self._pattern = pattern

        tubedict = {
            tube: self._inpath / "tube{}.csv".format(tube) for tube in [1 ,2]
        }

        self._data = InputData.from_files(tubedict, name=inpath, tubes=[1, 2])

    def apply_visualization(self, method):
        """Apply visualization specified in string."""
        dataname = self._pattern or self._inpath.stem
        output = self._outpath / "{}_{}".format(dataname, method)

        data, labels = self._data.split_data_labels(self._data.data)
        label, unique = pd.factorize(labels, sort=True)

        plot_creator(((output, method), (0, data, label, unique)))

        # with Pool(processes=6) as pool:
        #     pool.map(plot_creator, self._data.iter_add(output, method))


def main():
    args = parse_arguments()
    vis = Visualizer(args.input, args.output, args.pattern)
    vis.apply_visualization(args.method)


if __name__ == "__main__":
    main()
