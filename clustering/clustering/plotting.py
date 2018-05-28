from matplotlib.figure import Figure

def plot_overview(data, tube=1, coloring=None, title=None, s=1):
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
            ("CD19-APCE750", "CD22-PacBlue"),
            ("CD19-APCA750", "CD11c-PC7"),
            ("CD25-PC5.5", "CD11c-PC7"),
        ],
    }

    tube_channels = channels[tube]

    viewnum = len(tube_channels)

    h = int((viewnum-1) / 4) + 1
    w = viewnum if viewnum < 5 else 4

    fig = Figure(dpi=100, figsize=(4*w, 4*h))

    for i, (xchannel, ychannel) in enumerate(tube_channels):
        ax = fig.add_subplot(h, w, i+1)
        ax.scatter(data[xchannel], data[ychannel], s=s, c=coloring, marker=".")
        ax.set_xlabel(xchannel)
        ax.set_xlim(0, 1023)
        ax.set_ylabel(ychannel)
        ax.set_ylim(0, 1023)

    if title is None:
        title = "Tube {} overview".format(tube)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
