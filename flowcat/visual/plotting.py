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

import keras
import vis

from scipy.cluster import hierarchy

from ..dataset import case as ccase
from ..models import tfsom


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

def plot_tube(case, tube, grads, classes, sommappath=""):
    '''Plot scatterplots where the color indicates the maximum saliency value of the two
    corresponding channels.
    Args:
        case: Case object for which the gradients were computed.
        tube: Tube for which the values will be plotted.
        grads: Saliency gradients.
        classes: List of classes for which the salincy are supposed to be plotted.
        title: Main title for the subplots.
        sommappath: Path to the sommap data.
        plot_path: Path where the plot is supposed to be saved.
    Returns:
        List of figures (one for each class). This figures needs to be bound to a backend.
    '''
    tubecase = case.get_tube(tube)
    fcsdata = tubecase.data.data

    tubegates = ALL_VIEWS[tube]

    nodedata = map_fcs_to_sommap(case, tube, sommappath)

    fcsmapped, gridwidth = map_fcs_to_sommap(case, tube, sommappath)
    figures = []
    for idx, grad in enumerate(grads):
        chosen_selections = []
        for i, gating in enumerate(tubegates):
            channel_gradients = [grad[..., fcsdata.columns.get_loc(gate)] for gate in gating]
            max_gradients = np.maximum(channel_gradients[0], channel_gradients[1])
            mapped_gradients = [max_gradients[index] for index in fcsmapped['somnode']]
            fcsmapped['gradients'] = pd.Series(mapped_gradients, index=fcsmapped.index)
            chosen_selection = sommap_selection(fcsmapped.sort_values(by='gradients', ascending=False), max_gradients, gridwidth=gridwidth)
            chosen_selections.append(chosen_selection)
            fcsmapped.drop('gradients', 1, inplace=True)
        fig = Figure(figsize=(12, 8), dpi=300)
        for i, selection in enumerate(chosen_selections):
            axes = fig.add_subplot(3, 4, i + 1)
            draw_scatterplot(
                axes, fcsdata, tubegates[i], selections=selection)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.suptitle(f"Scatterplots Tube {tube} Class {classes[idx]}")
        figures.append(fig)
    return figures

def draw_saliency_heatmap(case, gradients, classes,  tube):
    '''Plots the saliency values as heatmeap for each of the classes.
    Args:
        case: Case object for which the gradients were computed.
        gradients: Saliency gradient values.
        classes: List of classes for which the heatmap will be plotted.
        tube: Tube which is supposed to be plotted.
    Returns:
        List of figures (one for each class). This figures needs to be bound to a backend.
    '''
    tubecase = case.get_tube(tube)
    channels = tubecase.data.data.columns
    figures = []
    for idx, group in enumerate(classes):
        #if saliency were not maximized a heatmap for each channel is plotted
        if gradients[tube-1][idx].ndim > 1:
            fig = Figure(figsize=(3, 4), dpi=300)
            for i in range(0, gradients[tube-1][idx].shape[-1]):
                axes = fig.add_subplot(3, 4, i + 1)
                axes.set_title(f"{channels[i]}", fontsize=5)
                axes.imshow(gradients[tube-1][idx][..., i].reshape(34, 34), cmap='jet')
                axes.tick_params(axis='both', labelsize=3,
                                 width=0.5, length=0.5)
        else:
            fig = Figure(figsize=(1, 1), dpi=300)
            axes = fig.add_subplot(111)
            axes.imshow(gradients[tube-1][idx].reshape(34, 34), cmap='jet')
        fig.tight_layout(pad=0.3)
        figures.append(fig)
    return figures

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

    if isinstance(data, ccase.FCSData):
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
        FigureCanvas(fig)
        fig.tight_layout()
        fig.savefig(str(filename), dpi=300)

    if dendroname is not None:
        fig = Figure()
        axes = fig.add_subplot(111)
        draw_dendrogram(axes, confusion_matrix, classes=classes)
        FigureCanvas(fig)
        fig.tight_layout()
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
                s=1, marker=".", c=[color], label=label)

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

def calc_saliency(dataset, case, model, classes, layer_idx=-1, maximization=False):
    '''Calculates the saliency values / gradients for the case, model and each of the classes.
    Args:
        dataset: SOMMapDataset object.
        case: Case object for which the saliency values will be computed.
        model: Path to hd5 file containing a keras model.
        layer_idx: Index of the layer for which the saleincy values will be computed.
        maximization: If true, the maximum of the saliency values over all channels will be returned.
    Returns:
        List of gradient values sorted first by tube and then class (e.g. [[tube1_class1,tube1_class1][tube2_class1,tube2_class2]]).
    '''
    # load existing model
    model = keras.models.load_model(model)

    # modify model for saliency usage
    model.layers[layer_idx].activation = keras.activations.linear
    model = vis.utils.utils.apply_modifications(model)

    xdata, ydata = dataset.get_batch_by_label(case.id)

    input_indices = [*range(len(xdata))]

    gradients = [vis.visualization.visualize_saliency(model, layer_idx, dataset.groups.index(
        group), seed_input=xdata, input_indices=input_indices, maximization=maximization) for group in set(classes)]


    if maximization:
        # regroup gradients into tube1 and tube2
        gradients = [[grad[0].flatten() for grad in gradients], [
            grad[1].flatten() for grad in gradients]]
    else:
        # regroup gradients into 2D array (nodes,channels) for tube1 and tube2
        gradients = [[grad[0].reshape(1156, 12) for grad in gradients], [
            grad[1].reshape(1156, 12) for grad in gradients]]

    return gradients

def map_fcs_to_sommap(case, tube, sommap_path):
    """Map for the given case the fcs data to their respective sommap data."""
    sommap_data = get_sommap_tube(sommap_path, case.id, tube)
    counts = sommap_data["counts"]
    sommap_data.drop(["counts", "count_prev"],
                     inplace=True, errors="ignore", axis=1)
    gridwidth = int(np.round(np.sqrt(sommap_data.shape[0])))

    tubecase = case.get_tube(tube)
    # get scaled zscores
    fcsdata = tubecase.get_data().data

    model = tfsom.TFSom(
        m=gridwidth, n=gridwidth, channels=sommap_data.columns,
        initialization_method="reference", reference=sommap_data,
        max_epochs=0, tube=tube)

    model.train([fcsdata], num_inputs=1)
    mapping = model.map_to_nodes(fcsdata)

    fcsdata["somnode"] = mapping
    return fcsdata, gridwidth


def get_sommap_tube(dataset_path, label, tube):
    '''Read the sommap data at the given path.'''
    path = dataset_path / f"{label}_t{tube}.csv"
    data = pd.read_csv(path, index_col=0)
    return data


def sommap_selection(fcsdata, grads, gridwidth=32):
    '''Determine a color for a somnode based on the corresponding gradient value.'''
    selection = []
    grad_colors = cm.ScalarMappable(cmap='autumn').to_rgba(1 - grads)
    for name, gdata in fcsdata.groupby("somnode"):
        color = grad_colors[name]
        if grads[name] < 0.1:
            color = [0.95, 0.95, 0.95, 0.1]
        else:
            color[3] = grads[name]
        selection.append((gdata.index, color, name))
    return selection
