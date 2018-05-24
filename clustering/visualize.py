import math
import logging
from itertools import combinations
import os
import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import cluster
from sklearn.pipeline import Pipeline

import fcsparser
from FlowCytometryTools import graph
from FlowCytometryTools.core import transforms

from clustering.clustering import create_pipeline, FCSLogTransform


PLOTDIR = "plots"

logging.basicConfig(level=logging.DEBUG)

RE_CHANNEL = re.compile(r"\$P\d+S")

def print_meta(metadata):
    machine = metadata["$CYT"]
    date = metadata["$DATE"]
    starttime = metadata["$BTIM"]
    endtime = metadata["$ETIM"]
    print("{} -- {} {}->{}".format(machine, date, starttime, endtime))

    channels = [
        metadata[m[0]] for m in [RE_CHANNEL.match(k) for k in metadata] if m
    ]
    channelinfo = {}
    for i, channel in enumerate(channels):
        colrange = metadata["$P{}R".format(i+1)]
        colexpo = metadata["$P{}E".format(i+1)]
        f1, f2 = colexpo.split(",")
        channelinfo[channel] = {
            "range": float(colrange),
            "f1": float(f1),
            "f2": float(f2),
        }
    print(" Channels:\n   {}".format("\n   ".join(channels)))
    return channelinfo


def print_data(data):
    print(data)


def create_cluster_pipe():
    return Pipeline(steps=[
        # ("scale", StandardScaler()),
        ("clust", cluster.DBSCAN(eps=200, min_samples=500)),
    ])


def plot_data(path, data, predictions, channels):
    fig = Figure(dpi=300, figsize=(10,10))
    FigureCanvas(fig)
    # autosize grid
    r = math.ceil(math.sqrt(len(channels) + 1))
    for i, (xchannel, ychannel) in enumerate(channels):
        ax = fig.add_subplot(r, r, i+1)
        ax.scatter(data[xchannel], data[ychannel], s=1, c=predictions, marker=".")
        ax.set_xlabel(xchannel)
        ax.set_ylabel(ychannel)
    # cax = fig.add_subplot(r, r, len(channels)+1)
    # colorbar = matplotlib.colorbar.ColorbarBase(cax, orientation="horizontal")
    fig.tight_layout()
    fig.savefig(path)


def process_fcs(filepath, label, plotdir, channels):
    view_channels = list(combinations(channels, 2))

    meta, data = fcsparser.parse(filepath, encoding="latin-1")
    channel_info = print_meta(meta)

    pipe = create_cluster_pipe()

    res = pipe.fit_predict(data[channels].values)
    plot_data(os.path.join(plotdir, label+"_kmeans"), data, res, view_channels)

    # pipe = create_pipeline()
    # pipe.fit(data[channels])

    # weights = pipe.named_steps["clust"].model.output_weights
    # df_weights = pd.DataFrame(weights)
    # df_weights.columns = sel_chans
    # df_weights["predictions"] = df_weights.index

    # predictions = pipe.predict(data[sel_chans])
    # data["predictions"] = predictions


    # os.makedirs(plotdir, exist_ok=True)
    # plot_data(os.path.join(plotdir, label), data, view_channels)

    # plot_data(os.path.join(plotdir, label+"_nodes"), df_weights, view_channels)



input_file = "data/18-000002-PB CLL 9F 01 N17 001.LMD"

# sel_chans = ["SS INT LIN", "CD45-KrOr", "FS INT LIN", "CD19-APCA750"]
sel_chans = ["CD45-KrOr", "SS INT LIN"]

channels = [
    ("SS INT LIN", "FS INT LIN"),
    ("CD45-KrOr", "SS INT LIN"),
    ("CD19-APCA750", "SS INT LIN"),
]
# channels = [
#     ("CD45-KrOr"),
#     ("SS INT LIN"),
# ]

process_fcs(input_file, "simple_test_ss_cd45", PLOTDIR, sel_chans)

# data = FCSLogTransform().transform(data)
# def logTransform(data):
#     transCols = [c for c in data.columns if "LIN" not in c]
#     for col, f in channel_info.items():
#         if f["f1"]:
#             data[col] = 10 ** (f["f1"] * data[col] / f["range"]) * f["f2"]
#     print(transCols)
#     print(data.min(axis=0), data.max(axis=0))
#
#     data[transCols] = np.log10(data[transCols])
#     return data

# data = logTransform(data)
