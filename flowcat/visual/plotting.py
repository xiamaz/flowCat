'''
Learning visualization functions
'''
import os
import logging
import itertools
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib import cm

import seaborn
from scipy.cluster import hierarchy

from ..data.case import FCSData


LOGGER = logging.getLogger(__name__)


ALL_VIEWS = {
    1: [
        ["CD19-APCA750", "CD79b-PC5.5"],
        ["CD19-APCA750", "CD5-PacBlue"],
        ["CD20-PC7", "CD23-APC"],
        ["CD19-APCA750", "CD10-PE"],
        ["CD19-APCA750", "FMC7-FITC"],
        ["CD20-PC7", "CD5-PacBlue"],
        ["CD19-APCA750", "IgM-ECD"],
        ["CD10-PE", "FMC7-FITC"],
        ["SS INT LIN", "FS INT LIN"],
        ["CD45-KrOr", "SS INT LIN"],
        ["CD19-APCA750", "SS INT LIN"],
    ],
    2: [
        ["CD19-APCA750", "Lambda-PE"],
        ["CD19-APCA750", "Kappa-FITC"],
        ["Lambda-PE", "Kappa-FITC"],
        ["CD19-APCA750", "CD22-PacBlue"],
        ["CD19-APCA750", "CD103-APC"],
        ["CD19-APCA750", "CD11c-PC7"],
        ["CD25-PC5.5", "CD11c-PC7"],
        ["Lambda-PE", "Kappa-FITC"],
        ["SS INT LIN", "FS INT LIN"],
        ["CD45-KrOr", "SS INT LIN"],
        ["CD19-APCA750", "SS INT LIN"],
    ],
    3: [
        ["CD3-ECD", "CD4-PE"],
        ["CD3-ECD", "CD8-FITC"],
        ["CD4-PE", "CD8-FITC"],
        ["CD56-APC", "CD3-ECD"],
        ["CD4-PE", "HLA-DR-PacBlue"],
        ["CD8-FITC", "HLA-DR-PacBlue"],
        ["CD19-APCA750", "CD3-ECD"],
        ["SS INT LIN", "FS INT LIN"],
        ["CD45-KrOr", "SS INT LIN"],
        ["CD3-ECD", "SS INT LIN"]
    ]
}


def save_figure(figure, path):
    """Save a figure to the specified path after adding it to the default Agg
    canvas.

    Args:
        figure: Figure object.
        path: Output path.
    """
    FigureCanvas(figure)
    figure.savefig(path)


def plot_histogram3d_som(sommap, count_column="counts"):
    """Plot a 3d histogram for the given sommap."""
    coorddata = sommap_to_coorddata(sommap, count_column)
    fig = Figure(figsize=(8, 8), dpi=300)

    axes = fig.add_subplot(111, projection="3d")

    draw_histogram_3d(axes, coorddata)

    axes.set_title(f"Distribution of {count_column}")
    fig.tight_layout()
    return fig


def plot_heatmap_som(sommap, count_column="counts", cmap=cm.Blues):  # pylint: disable=no-member
    """Plot heatmap for som weights data."""
    xydata = sommap_to_xydata(sommap, data_columns=count_column)

    fig = Figure(figsize=(8, 8), dpi=300)
    axes = fig.add_subplot(111, projection="2d")
    axes.imshow(xydata, interpolation='nearest', cmap=cmap)

    return fig


def plot_colormap_som():
    """Plot colormap for given SOM data."""
    pass


def plot_scatterplot(data, tube, selections=None, selected_views=None, horiz_width=4):
    """Plot a scatterplot for given data.
    Args:
        data: FCS or SOMmap.
        tube: Tube of origin for the given data.
        selections: Optional list of (index, color, label) tuples for colored plotting.
        selected_views: Optional list of (channelx, channely) tuples for specific scatterplots.
        horiz_width: Number of scatterplots in a row.
    Returns:
        Figure containing drawn axes. This figure needs to be bound to a backend.
    """
    if selected_views is None:
        selected_views = ALL_VIEWS[tube]

    if isinstance(data, FCSData):
        ranges = data.ranges
        data = data.data
    else:
        ranges = None

    vert_width = int(np.ceil(len(selected_views) / horiz_width))

    fig = Figure(figsize=(horiz_width * 4, vert_width * 4), dpi=300)
    for i, channels in enumerate(selected_views):
        if ranges is not None:
            cranges = ranges.loc[["min", "max"], channels].values.transpose().tolist()
        else:
            cranges = ranges
        axes = fig.add_subplot(vert_width, horiz_width, i + 1)
        axes = draw_scatterplot(axes, data, channels=channels, selections=selections, ranges=cranges)

    fig.suptitle(f"Tube {tube}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_confusion_matrix(
        confusion_matrix, classes,
        normalize=False,
        title='Confusion matrix',
        cmap=cm.Blues,  # pylint: disable=no-member
        filename='confusion.png',
        dendroname="dendro.png",
        sizes=None, colorbar=False
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    Args:
        confusion_matrix:
        classes:
        normalize:
        title:
        cmap:
        filename:
    """
    if normalize:
        LOGGER.debug("Normalized confusion matrix.")
        confusion_matrix = normalize_confusion_matrix(confusion_matrix)
    else:
        LOGGER.debug('Confusion matrix, without normalization')
        confusion_matrix = confusion_matrix.astype('int')

    if filename is not None:
        fig = Figure()
        axes = fig.add_subplot(111)
        axes.set_title(title)
        fig.tight_layout()
        FigureCanvas(fig)
        fig.savefig(str(filename), dpi=300)

    if dendroname is not None:
        fig = Figure()
        axes = fig.add_subplot(111)
        draw_dendrogram(axes, confusion_matrix, classes=classes)
        fig.tight_layout()
        FigureCanvas(fig)
        fig.savefig(dendroname, dpi=300)


def draw_confusion_matrix(
        axes, confusion_matrix,
        classes=None, sizes=None,
        cmap=cm.Blues,  # pylint: disable=no-member
):
    """Draw a confusion matrix on the given axes.
    Args:
        axes:
        confusion_matrix:
        classes:
    Returns:
        Axes containing confusion matrix.
    """
    if classes is None and isinstance(confusion_matrix, pd.DataFrame):
        classes = list(confusion_matrix.columns)
    else:
        raise RuntimeError("Cannot infer groups from numpy array.")

    if isinstance(pd.DataFrame):
        confusion_matrix = confusion_matrix.values

    # set tick mark values and confusion matrix values for image plot
    tick_marks = np.arange(len(classes), dtype=float)
    confusion_img = confusion_matrix.copy()
    if sizes is not None:
        for i, size in enumerate(sizes):
            if size > 1:
                # shift the tickmarks so that they are in the center of larger
                # merged cells
                tick_marks[i] = np.arange(tick_marks[i], size + tick_marks[i], dtype=float).mean()
                tick_marks[i+1:] = tick_marks[i+1:] + size - 1

        confusion_img = matrix_to_sizes(confusion_matrix, sizes)

    axes.imshow(confusion_img, interpolation='nearest', cmap=cmap)
    axes.set_xticks(tick_marks)
    axes.set_xticklabels(classes)
    axes.set_yticks(tick_marks)
    axes.set_yticklabels(classes)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')

    # print the number values in each cell
    if "float" in confusion_img.dtype.name:
        fmt = ".2f"
    else:
        fmt = "d"
    thresh = confusion_img.max() / 2.
    for i, j in itertools.product(range(len(tick_marks)),
                                  range(len(tick_marks))):
        axes.text(tick_marks[j], tick_marks[i], format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    return axes


def draw_dendrogram(axes, confusion_matrix, classes=None):
    """Plot a dendrogram based on hierarchical clustering of the input
    confusion matrix.
    Args:
        axes: Axes used for plotting.
        confusion_matrix: Pandas dataframe or numpy array.
        classes: Labels to be used for plotting. Required for numpy array.
            Optional for dataframes. Will be inferred from column names if not
            given for the latter.
    Returns:
        Plotted axes.
    """
    if classes is None and isinstance(confusion_matrix, pd.DataFrame):
        classes = list(confusion_matrix.columns)
    else:
        raise RuntimeError("Cannot infer groups from numpy array.")

    pairwise_dists = hierarchy.distance.pdist(
        confusion_matrix, metric='euclidean')
    dist_linkage = hierarchy.linkage(pairwise_dists, method='single')

    hierarchy.dendrogram(
        dist_linkage, show_contracted=True, labels=classes, ax=axes)
    return axes


def draw_scatterplot(axes, data, channels, selections=None, ranges=None):
    """Draw a scatterplot on the given axes.
    Args:
        data: Pandas dataframe containing channels.
        channels: X and Y channels for plotting.
        axes: Matplotlib axes for plotting.
        selections: List of tuples of selection and color for coloured plotting.
    Returns:
        Plotted axes.
    """
    xchannel, ychannel = channels
    x = data[xchannel]
    y = data[ychannel]

    if selections is None:
        axes.scatter(x, y, s=1, marker=".")
    else:
        for sel, color, label in selections:
            axes.scatter(
                data.loc[sel, xchannel], data.loc[sel, ychannel],
                s=1, marker=".", c=color, label=label)

    axes.set_xlabel(xchannel)
    axes.set_ylabel(ychannel)

    if ranges is not None:
        rangex, rangey = ranges
    else:
        rangex, rangey = ((0, 1023), (0, 1023))

    axes.set_xlim(*rangex)
    axes.set_ylim(*rangey)
    return axes


def draw_histogram_3d(axes, data):
    """Draw 3D histogram using the given coordinate data.

    Args:
        data: Coordinate data created with sommap_to_xydata.
        axes: 3D plottable axis created with projection='3d'
    Returns:
        Axes with 3D histogram.
    """
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = np.zeros_like(x)

    dx = 1 * np.ones_like(x)  # width in x dimension
    dy = dx.copy()  # width in y dimension
    dz = data.iloc[:, 2]
    axes.bar3d(
        x, y, z,
        dx, dy, dz
    )
    axes.view_init(elev=45, azim=45)
    return axes


def normalize_confusion_matrix(confusion_matrix):
    """Normalize the given confusion matrix."""
    classes = None
    if isinstance(confusion_matrix, pd.DataFrame):
        classes = confusion_matrix.columns
        confusion_matrix = confusion_matrix.values
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    if classes is not None:
        return pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    return confusion_matrix


def matrix_to_sizes(matrix, sizes):
    """Expand the confusion matrix to the given sizes by duplicating rows and
    columns.

    Args:
        matrix: Numpy array.
        sizes: List of sizes for each column in the original data.
    Returns:
        Expanded matrix.
    """
    # tile horizontally
    hreplicated = []
    for i, size in enumerate(sizes):
        hreplicated.append(np.tile(matrix[[i], :], (size, 1)))
    matrix = np.concatenate(hreplicated)

    # tile vertically
    vreplicated = []
    for i, size in enumerate(sizes):
        vreplicated.append(np.tile(matrix[:, [i]], (1, size)))
    matrix = np.concatenate(vreplicated, axis=1)
    return matrix


def sommap_to_coorddata(data, data_column="counts"):
    """Get counts from dataframe together with coordinates for plotting.
    Args:
        data: Pandas dataframe with counts. Should be from square SOM
        count_column: Name of count column.
    Returns:
        Pandas dataframe with x, y coordinates and z as value.
    """
    gridsize = int(np.round(np.sqrt(data.shape[0])))
    coords = pd.DataFrame(
        list(itertools.product(range(gridsize), range(gridsize)))
    )
    coords[data_column] = data[data_column]
    return coords


def sommap_to_xydata(data, data_columns="counts"):
    """Reshape input into coordinate from with x in rows and y in columns.
    Args:
        data: Pandas dataframe to be reshaped.
        data_columns: Single string column or a list of columns to be used.
    Returns:
        Reshaped numpy matrix containing data columns as last dimension.
    """

    gridsize = int(np.round(np.sqrt(data.shape[0])))
    xydata = data[data_columns].values.reshape(shape=(gridsize, gridsize, -1))
    return xydata
