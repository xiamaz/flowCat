"""
Create simple plots for SOMs and other objects.
"""
import logging
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch

LOGGER = logging.getLogger(__name__)


def scale_weights_to_colors(weights):
    """Scale all given channels similar to tf.summary.image:
    if all positive:
        rescale to largest 1.0
    else:
        TODO: not implemented yet
        shift 0.0 to 0.5 and rescale to either min at 0 or max at 255
    """
    minimum = np.min(weights)
    if minimum < 0:
        LOGGER.warning(
            "Negative weights will not be transformed identically to tensorflow"
        )
    scaled = (weights - minimum) / (np.max(weights) - minimum)
    return scaled


def plot_color_grid(griddata, channels, scale=True):
    """Plot array with 3 color channels as RGB values."""
    if scale:
        griddata = scale_weights_to_colors(griddata)

    fig = Figure()
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.1])
    axes = fig.add_subplot(grid[0, 0])
    axes.imshow(griddata, interpolation="nearest")
    axes.set_axis_off()

    patches = [
        Patch(color=color, label=channel)
        for channel, color in zip(channels, ("red", "green", "blue"))
    ]

    ax_legend = fig.add_subplot(grid[1, 0])
    ax_legend.legend(handles=patches, framealpha=0.0, ncol=3, loc="center")
    ax_legend.set_axis_off()

    # Saving the figure
    FigureCanvas(fig)
    fig.tight_layout()
    return fig


def plot_som_grid(somdata, channels=None):
    """Plot a single som object with same visuals as tf.summary.image.
    """
    imgdata = somdata.data
    if channels is None:
        channels = somdata.markers[:3]

    indexes = [somdata.markers.index(c) if c is not None else None for c in channels]

    arr = np.stack([
        imgdata[:, :, i] if i is not None else np.zeros(imgdata.shape[:2])
        for i in indexes
    ], axis=-1)
    return plot_color_grid(arr, channels, scale=True)
