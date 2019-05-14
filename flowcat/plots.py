"""
Create simple plots for SOMs and other objects.
"""
from . import som

import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def plot_som_grid(somdata, destpath, channels=None):
    """Plot a single som object."""
    imgdata = somdata.data
    if channels is None:
        channels = imgdata.columns.values[:3]
    marker_data = imgdata[channels]
    arr = marker_data.values
    scaled = (arr - np.min(arr, axis=0)) / (np.max(arr, axis=0) - np.min(arr, axis=0))
    scaled = np.reshape(scaled, (32, 32, 3))

    fig = Figure()
    axes = fig.add_subplot(111)
    axes.imshow(scaled, interpolation="nearest")

    # Saving the figure
    FigureCanvas(fig)
    fig.tight_layout()
    fig.savefig(str(destpath))
