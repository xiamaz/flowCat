"""Create plots for training history."""
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def plot_history(history_data: dict, title="") -> Figure:
    """Create a line plot with the given history data over epochs."""
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    for name, data in history_data.items():
        ax.plot(data, legend=name)

    ax.legent()
    fig.tight_layout()
    return fig
