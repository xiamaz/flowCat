import sys

import logging

import pandas as pd
import networkx as nx
import scipy as sp

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from clustering.transformation import tfsom
from clustering.collection import CaseCollection

sys.path.append("../report")
from report import base as rbase


GROUPS = [
    # "CLL", "CLLPL", "FL", "HZL", "LPL", "MBL", "Mantel", "Marginal", "normal"
    "CLL", "MBL", "normal"
]


def plot_node_weights(weights, gridwidth=10):
    """Create 2d plot of node weights."""
    print(weights)
    nodenum, channels = weights.shape

    gridheight = int(nodenum / gridwidth)

    grid = nx.grid_2d_graph(gridwidth, gridheight)
    # relabel nodes to continuous mapping
    mapping = {
        n: n[0] * gridwidth + n[1] for n in grid
    }
    grid = nx.relabel_nodes(grid, mapping)

    # add distance to edges
    for edge in grid.edges:
        fromdata = weights.iloc[edge[0], :]
        todata = weights.iloc[edge[1], :]

        euc_dist = sp.spatial.distance.euclidean(fromdata.values, todata.values)
        grid.edges[edge]["weight"] = euc_dist

    plt.figure()
    nx.draw_kamada_kawai(grid, with_labels=True)
    plt.savefig("grid_euc_5")


def plot_weight_2d_labeled(weights):
    """Create 2d scatterplots with labels."""
    views = [
        ("CD45-KrOr", "SS INT LIN"),
        ("Kappa-FITC", "Lambda-PE"),
    ]
    fig = plt.figure(figsize=(10, 5))
    for i, (x, y) in enumerate(views):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.scatter(weights[x], weights[y])

        for name, row in weights.iterrows():
            ax.annotate(name, (row[x], row[y]))

        ax.set_xlabel(x)
        ax.set_ylabel(y)

    plt.savefig("grid_scatter_5")


def configure_print_logging(rootname="clustering"):
    """Configure default logging for visual output to stdout."""
    rootlogger = logging.getLogger(rootname)
    rootlogger.setLevel(logging.INFO)
    formatter = logging.Formatter()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)


def main():
    configure_print_logging()

    cases = CaseCollection(inputpath="s3://mll-flowdata/CLL-9F", tubes=[1, 2])

    basic_train = cases.create_view(groups=GROUPS, num=5, infiltration=10.0)

    t_data = basic_train.get_tube(2)

    # basic_test = cases.create_view(groups=GROUPS, num=100)

    fitdata = pd.concat([t.data for t in t_data.data])

    classifier = tfsom.SelfOrganizingMap(10, 10, max_epochs=2)
    classifier.fit(fitdata)

    plot_node_weights(classifier.weights)


if __name__ == "__main__":
    main()
