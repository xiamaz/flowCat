from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.figure import Figure
from matplotlib.artist import setp

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

    h = int((viewnum-1) / 4) + 1
    w = viewnum if viewnum < 5 else 4

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

    axes = []
    for i, (xchannel, ychannel) in enumerate(tube_channels):
        if inner_grid:
            ax = fig.add_subplot(inner_grid[i])
        else:
            ax = fig.add_subplot(h, w, i+1)
        ax.scatter(data[xchannel], data[ychannel], s=s, c=coloring, marker=".")
        ax.set_xlabel(xchannel)
        ax.set_xlim(0, 1023)
        ax.set_ylabel(ychannel)
        ax.set_ylim(0, 1023)
        axes.append(ax)

    if title is None:
        title = "Tube {} overview".format(tube)
    # setp(axes, title=title)
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
    return fig, (w*4, h*4)
