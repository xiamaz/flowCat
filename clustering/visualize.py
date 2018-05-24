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

from clustering.clustering import create_pipeline, FCSLogTransform
from clustering.case_collection import CaseCollection


PLOTDIR = "plots_test"

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
        ("clust", cluster.DBSCAN(eps=30, min_samples=200, metric="manhattan")),
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


def get_node_position(data, res, channels, position):
    (c1, c2) = channels
    (p1, p2) = position
    x2 = 1023 if p1 == "+" else 0
    y2 = 1023 if p2 == "+" else 0
    closest = None
    cdist = None
    for i in np.unique(res):
        if i == -1:
            continue
        means = data.loc[res == i, channels].mean(axis=0)
        print(i, means)
        dist = math.sqrt((x2-means[c1])**2+(y2-means[c2])**2)
        print("Distance :", dist, "Old :", cdist)
        if closest is None or cdist > dist:
            closest = i
            cdist = dist
    print("Selected", closest)
    sel_res = np.zeros(res.shape)

    sel_res[res == closest] = 1
    print(sel_res)
    return sel_res

def process_fcs(filepath, label, plotdir, channels, pos):
    print("Processing", label)
    os.makedirs(plotdir, exist_ok=True)
    view_channels = list(combinations(channels, 2))

    meta, data = fcsparser.parse(filepath, encoding="latin-1")
    channel_info = print_meta(meta)

    pipe = create_cluster_pipe()

    res = pipe.fit_predict(data[channels].values)
    sel_res = get_node_position(data, res, channels, pos)
    plot_data(os.path.join(plotdir, label+"_kmeans"), data, sel_res, view_channels)
    print("==========================")

    # pipe = create_pipeline()
    # pipe.fit(data[channels])

    # weights = pipe.named_steps["clust"].model.output_weights
    # df_weights = pd.DataFrame(weights)
    # df_weights.columns = sel_chans
    # df_weights["predictions"] = df_weights.index

    # predictions = pipe.predict(data[sel_chans])
    # data["predictions"] = predictions


    # plot_data(os.path.join(plotdir, label), data, view_channels)

    # plot_data(os.path.join(plotdir, label+"_nodes"), df_weights, view_channels)


# coll = CaseCollection("case_info.json", "mll-flowdata", tmpdir="tests/tube2")

# for label, group, fcsdata in coll.get_all_data(num=1, tube=2):
#     print(label, group)

sel_chans = ["CD45-KrOr", "SS INT LIN"]
position = ["+", "-"]

indir = "tests/tube1"
for cohort in os.listdir(indir):
    for i, filename in enumerate(os.listdir(os.path.join(indir, cohort))):
        filepath = os.path.join(indir, cohort, filename)
        process_fcs(filepath, "{}_{}".format(cohort, i), PLOTDIR, sel_chans, position)

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
