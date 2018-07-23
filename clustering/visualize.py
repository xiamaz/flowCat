"""Visualization of separation in output data of classification."""
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm


def parse_arguments():
    """Get paths and desires processing method from command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("input", help="CSV input data.")
    parser.add_argument("method", help="Analysis method.", type=str.lower)
    parser.add_argument("output", help="Output directory for plots.")

    args = parser.parse_args()
    return args


def load_data(path):
    """Load csv data into a pandas dataframe."""
    return pd.read_table(path, sep=",", index_col=0)


class Plot:
    """Basic abstraction for plotting."""
    def __init__(self, data):
        self._data = data[[c for c in data.columns if c.isdigit()]]
        self._labels, self._uniques = pd.factorize(data["group"])

    def save(self, path):
        """Save the created plot at the given location.
        Automatically create any missing intermediary plot directories.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

    def scatterplot(self, transformed, ax):
        colors = [
            cm.Set2(float(i) / len(self._uniques))
            for i in range(len(self._uniques))
        ]

        for i, unique in enumerate(self._uniques):
            selection = transformed[self._labels == i]
            ax.scatter(
                selection[:, 0], selection[:, 1], c=colors[i], label=unique,
                s=1, marker="."
            )
        ax.legend(markerscale=10)
        return ax


class PCAPlot(Plot):
    """Use principal component analysis to linearly analyse our data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = PCA(n_components=2)

    def save(self, path):
        super().save(path)

        transformed = self._model.fit_transform(self._data)
        fig = Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        self.scatterplot(transformed, ax)
        ax.set_title("PCA in 2 dimensions")

        FigureCanvas(fig)
        fig.savefig(str(path))


class TSNEPlot(Plot):
    """Create a tSNE plot of the given data. Will be very slow for high
    dimensional data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = TSNE(n_components=2)

    def save(self, path):
        super().save(path)

        transformed = self._model.fit_transform(self._data)
        fig = Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        self.scatterplot(transformed, ax)
        ax.set_title("tSNE in 2 dimensions")

        FigureCanvas(fig)
        fig.savefig(str(path))


class Visualizer:
    def __init__(self, inpath, outpath):
        self._inpath = Path(inpath)
        self._outpath = Path(outpath)

        self._data = load_data(self._inpath)

    def apply_visualization(self, method):
        """Apply visualization specified in string."""
        dataname = self._outpath.stem
        if method == "pca":
            plot = PCAPlot(self._data)
            plot.save(self._outpath / "{}_pca".format(dataname))
        elif method == "tsne":
            plot = TSNEPlot(self._data)
            plot.save(self._outpath / "{}_tsne".format(dataname))
        else:
            raise RuntimeError("Unknown method {}".format(method))


def main():
    args = parse_arguments()
    vis = Visualizer(args.input, args.output)
    vis.apply_visualization(args.method)


if __name__ == "__main__":
    main()
