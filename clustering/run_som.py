import sys
import pathlib

import logging

import pandas as pd
import networkx as nx
import scipy as sp

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from clustering.transformation import tfsom, fcs, pregating
from clustering.collection import CaseCollection


GROUPS = [
    "CLL", "PL", "FL", "HCL", "LPL", "MBL", "MCL", "MZL", "normal"
]


def plot_node_weights(weights, gridwidth=10, path="gridplot"):
    """Create 2d plot of node weights."""
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

    fig = Figure()
    axes = fig.add_subplot(111)
    nx.draw_kamada_kawai(grid, with_labels=True, ax=axes)
    FigureCanvas(fig)
    fig.savefig(path)


def plot_weight_2d_labeled(
        weights, selection=None, path="2dplot", annot=False
):
    """Create 2d scatterplots with labels."""
    views = [
        ("CD45-KrOr", "SS INT LIN"),
        ("Kappa-FITC", "Lambda-PE"),
    ]
    fig = Figure(figsize=(10, 5))

    if annot:
        scargs = {"s": 1, "marker": "o"}
    else:
        scargs = {"s": 1, "marker": "."}

    for i, (x, y) in enumerate(views):
        ax = fig.add_subplot(1, 2, i + 1)
        if selection is None:
            ax.scatter(weights[x], weights[y], **scargs)
        else:
            unsel = weights.drop(selection)
            ax.scatter(unsel[x], unsel[y], color="g", **scargs)
            ax.scatter(selection[x], selection[y], color="r", **scargs)

        if annot:
            for name, row in weights.iterrows():
                ax.annotate(name, (row[x], row[y]))

        ax.set_xlabel(x)
        ax.set_ylabel(y)

    FigureCanvas(fig)
    fig.savefig(path)


def configure_print_logging(rootname="clustering"):
    """Configure default logging for visual output to stdout."""
    rootlogger = logging.getLogger(rootname)
    rootlogger.setLevel(logging.INFO)
    formatter = logging.Formatter()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)


def train_plot_data(tubedata, classifier, path, gridwidth, transargs):
    """Plot the tube data using the instantiated classification model."""
    preprocess = Pipeline(
        steps=[
            ("pregate", pregating.SOMGatingFilter(
                **(transargs if transargs is not None else {})
            ))
        ]
    )
    transdata = []

    transplot = path / "gates"
    transplot.mkdir(parents=True, exist_ok=True)

    for case in tubedata.data:
        transformed = preprocess.fit_transform(case.data)
        transdata.append(transformed)
        plot_weight_2d_labeled(
            case.data,
            transformed,
            path=transplot / "{}_{}_diff".format(
                case.parent.group, case.parent.id
            )
        )

    fitdata = pd.concat(
        transdata
    )

    classifier.fit(fitdata)

    # plot_node_weights(
    #     classifier.weights,
    #     path=str(path) + "_weights",
    #     gridwidth=gridwidth
    # )
    plot_weight_2d_labeled(classifier.weights, path=str(path) + "_2d_scatter")


def model_gradient(path, cases, *args, **kwargs):
    """Vary parameters in the classifier and generate different plots."""
    base_params = {
        "m": 10,
        "n": 10,
        "max_epochs": 3
    }

    modelclass= tfsom.SelfOrganizingMap


    param = "pregatingsize"
    for val in [3, 5, 10]:
        cur_params = base_params
        # cur_params = {
        #     **base_params, **{
        #         "m": val,
        #         "n": val,
        #     }
        # }
        model = modelclass(**cur_params)

        basic_train = cases.create_view(
            groups=GROUPS, num=val, infiltration=10.0
        )

        tubedata = basic_train.get_tube(2)

        plotpath = path / "{}_{}".format(param, val)
        train_plot_data(
            classifier=model,
            path=plotpath,
            gridwidth=cur_params["n"],
            tubedata=tubedata,
            transargs={
                "somargs": {"m": val, "n": val}
            }
        )


def simple_run(path, data, reference):
    """Very simple SOM run for tensorflow testing."""
    # only use data from the second tube
    tubedata = data.get_tube(2)

    marker_list = tubedata.data[0].markers

    def data_generator():
        for tdata in tubedata.data:
            lmddata = tdata.data[marker_list]
            yield lmddata

    print(reference)

    model = tfsom.TFSom(
        m=10,
        n=10,
        dim=len(marker_list),
        batch_size=5,
        max_epochs=5,
        reference=reference,
        initialization_method="reference",
    )

    print(len(tubedata.data))

    model.train(data_generator, num_inputs=len(tubedata.data))

    weights = model.output_weights
    print(weights.shape)
    # weights.to_csv("somweights.csv")


def main():
    configure_print_logging()

    cases = CaseCollection(inputpath="s3://mll-flowdata/CLL-9F", tubes=[1, 2])

    with open("labels.txt") as lfile:
        # remove newline character from each line
        simple_labels = [l.strip() for l in lfile]

    reference_weights = pd.read_csv("somweights.csv", index_col=0)

    plotpath = pathlib.Path("somplots/pregated")

    plotpath.mkdir(parents=True, exist_ok=True)


    simple_data = cases.create_view(labels=simple_labels)

    simple_run(path=plotpath, data=simple_data, reference=reference_weights)

    # model_gradient(cases=cases, path=plotpath)


if __name__ == "__main__":
    main()
