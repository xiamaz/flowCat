"""Visualization of separation in output data of classification."""
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


def load_data(path):
    """Load csv data into a pandas dataframe."""
    return pd.read_table(path, sep=",", index_col=0)


class Dataset:
    """Container for csv data."""

    def __init__(self, path, output, tube=None, pattern=""):
        self._outpath = output

        self._current = 0
        self._pattern = pattern
        self._tube = tube
        if path.is_dir():
            self._rdata = self._load_dir(path)
        else:
            self._rdata = [load_data(path)]

        self._data = [
            d[[c for c in d.columns if c.isdigit()]] for d in self._rdata
        ]
        self._labels = []
        self._uniques = None
        for data in self._rdata:
            label, unique = pd.factorize(data["group"], sort=True)
            self._labels.append(label)
            if self._uniques is None:
                self._uniques = unique

    @property
    def concat(self):
        return pd.concat(self._data) if len(self._data) > 1 else self._data[0]

    def _load_dir(self, path):
        files = path.glob("*{}*/tube{}.csv".format(self._pattern, self._tube))
        return [load_data(f) for f in files]

    def __iter__(self):
        self._current = 0

        # create the output directory
        self._outpath.parent.mkdir(parents=True, exist_ok=True)

        return self

    def __next__(self):
        if self._current >= len(self._data):
            raise StopIteration
        self._current += 1
        return (
            self._current,
            self._data[self._current - 1],
            self._labels[self._current - 1],
            self._uniques,
        )

    def iter_add(self, *args):
        for items in self:
            yield args, items


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


def create_plot(transfun, title, path, index, data, label, unique):
    tdata = transfun(data)

    plotpath = str(path) + "_{}".format(index)

    with new_plot(plotpath, title) as ax:
        scatterplot(tdata, label, unique, ax)


def plot_creator(args):
    (path, method), items = args
    dimensions = 2
    if method == "pca":
        model = PCA(n_components=dimensions)
        title = "PCA in {} dimensions".format(dimensions)
    elif method == "tsne":
        model = TSNE(n_components=dimensions)
        title = "tSNE in {} dimensions".format(dimensions)
    else:
        raise RuntimeError("Unknown method {}".format(method))

    create_plot(model.fit_transform, title, path, *items)


class Visualizer:
    def __init__(self, inpath, outpath, pattern):
        self._inpath = Path(inpath)
        self._outpath = Path(outpath)
        self._pattern = pattern

        self._data = Dataset(
            self._inpath, self._outpath, tube=1, pattern=pattern
        )

    def apply_visualization(self, method):
        """Apply visualization specified in string."""
        dataname = self._pattern or self._inpath.stem
        output = self._outpath / "{}_{}".format(dataname, method)

        with Pool(processes=6) as pool:
            pool.map(plot_creator, self._data.iter_add(output, method))


def main():
    args = parse_arguments()
    vis = Visualizer(args.input, args.output, args.pattern)
    vis.apply_visualization(args.method)


if __name__ == "__main__":
    main()
