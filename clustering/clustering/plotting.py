import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.figure import Figure
from matplotlib.artist import setp
from matplotlib import ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Maximum number of plots horizontally
MAX_PLOTWIDTH = 5

PLOTWIDTH = 6
PLOTHEIGHT = 4


# Basic plotting helpers
def create_plot(path, plotfunc):
    fig = Figure()
    fig = plotfunc(fig)
    FigureCanvas(fig)
    fig.savefig(path, dpi=100)


# elementary plot elements

def plot_histogram(data, ax, title=None):
    (mean_data, std_data) = data
    ax.plot(mean_data.index, mean_data, "k-")
    ax.fill_between(
        pd.to_numeric(mean_data.index),
        (mean_data-std_data).clip_lower(0),
        (mean_data+std_data).clip_upper(1.0)
    )
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.set_title(title)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Event distribution")
    return ax

def plot_2d(ax, channels, data, labels=None, dotsize=1):
    xchannel, ychannel = channels
    ax.scatter(data[xchannel], data[ychannel], s=dotsize, c=labels, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_xlim(0, 1023)
    ax.set_ylabel(ychannel)
    ax.set_ylim(0, 1023)
    return ax


# Complex combination plots

def plot_history(fig, history, title=""):
    num_plots = len(history)
    # calculate number of rows for plotting depending on plotwidth
    num_rows = int((num_plots-1) / MAX_PLOTWIDTH) + 1

    fig.set_size_inches(
        MAX_PLOTWIDTH * PLOTWIDTH,
        num_rows * PLOTHEIGHT,
    )

    for num, hist in enumerate(history):
        ax = fig.add_subplot(num_rows, MAX_PLOTWIDTH, num+1)
        ax = plot_2d(
            ax,
            ("CD45-KrOr", "SS INT LIN"),
            hist["data"], labels=hist["mod"]
        )
        ax.set_title(hist["text"])
    fig.suptitle("History {}".format(title))

    dims = [0, 0, 1, 0.95]
    fig.tight_layout(rect=dims)
    return fig

def plot_overview(
        data, tube=1, coloring=None, title=None, s=1,
        fig=None, gs=None, gspos=None
):
    """Overview plot over many channels.
    """
    channels = {
        1: [
            ("CD45-KrOr", "SS INT LIN"),
            ("CD19-APCA750", "SS INT LIN"),
            ("CD20-PC7", "CD23-APC"),
            ("CD10-PE", "CD19-APCA750"),
        ],
        2: [
            ("CD45-KrOr", "SS INT LIN"),
            ("CD19-APCA750", "SS INT LIN"),
            ("Kappa-FITC", "Lambda-PE"),
            ("CD19-APCA750", "CD22-PacBlue"),
            ("CD19-APCA750", "CD11c-PC7"),
            ("CD25-PC5.5", "CD11c-PC7"),
        ],
    }

    tube_channels = channels[tube]

    viewnum = len(tube_channels)

    h = int((viewnum-1) / MAX_PLOTWIDTH) + 1
    w = viewnum if viewnum < MAX_PLOTWIDTH else MAX_PLOTWIDTH

    if not fig:
        fig = Figure(dpi=100, figsize=(4*w, 4*h))
    if gs and gspos is not None:
        overview_grid = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[gspos], height_ratios=[1, 7]
        )

        inner_grid = GridSpecFromSubplotSpec(
            h, w, subplot_spec=overview_grid[1], wspace=0.4
        )
    else:
        inner_grid = None

    for i, channels in enumerate(tube_channels):
        if inner_grid:
            ax = fig.add_subplot(inner_grid[i])
        else:
            ax = fig.add_subplot(h, w, i+1)
        ax = plot_2d(ax, channels, data, labels=coloring, dotsize=s)

    if title is None:
        title = "Tube {} overview".format(tube)
    title_ax = fig.add_subplot(overview_grid[0])
    title_ax.text(
        0.5, 0.5, title,
        horizontalalignment='center',
        verticalalignment='center',
        size="x-large"
    )
    title_ax.axis("off")
    dims = [0, 0.03, 1, 0.95]
    if not inner_grid:
        fig.tight_layout(rect=dims)
    return fig, (w*PLOTWIDTH, h*PLOTHEIGHT)


def plot_upsampling(fig, data, title="Upsampling"):
    groups = data.groupby("group")
    for i, (group, gdata) in enumerate(groups):
        cdata = gdata.drop("infiltration", axis=1)
        mean_data = cdata.mean()
        std_data = cdata.std()
        ax = fig.add_subplot(5, 2, i+1)
        plot_histogram((mean_data, std_data), ax, title=group)
    fig.suptitle(title)
    fig.set_size_inches(20, 10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def get_umatrix(weights, m, n):
    """ Generates an n x m u-matrix of the SOM's weights.

    Used to visualize higher-dimensional data. Shows the average distance
    between a SOM unit and its neighbors.  When displayed, areas of a darker
    color separated by lighter colors correspond to clusters of units which
    encode similar information.
    :param weights: SOM weight matrix, `ndarray`
    :param m: Rows of neurons
    :param n: Columns of neurons
    :return: m x n u-matrix `ndarray`
    """
    umatrix = np.zeros((m * n, 1))
    # Get the location of the neurons on the map to figure out their neighbors.
    # I know I already have this in the SOM code but I put it here too to make
    # it easier to follow.
    neuron_locs = list()
    for i in range(m):
        for j in range(n):
            neuron_locs.append(np.array([i, j]))
    # Get the map distance between each neuron (i.e. not the weight distance).
    neuron_distmat = distance_matrix(neuron_locs, neuron_locs)

    for i in range(m * n):
        # Get the indices of the units which neighbor i
        neighbor_idxs = neuron_distmat[i] <= 1  # Change this to `< 2` if you want to include diagonal neighbors
        # Get the weights of those units
        neighbor_weights = weights[neighbor_idxs]
        # Get the average distance between unit i and all of its neighbors
        # Expand dims to broadcast to each of the neighbors
        umatrix[i] = distance_matrix(np.expand_dims(weights[i], 0), neighbor_weights).mean()

    return umatrix.reshape((m, n))


def plot_weights(weights, m, n):
    umatrix = get_umatrix(weights, m, n)
    fig = plt.figure()
    plt.imshow(umatrix, origin='lower')
    plt.show(block=True)
