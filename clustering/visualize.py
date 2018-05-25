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

from clustering.clustering import create_pipeline, FCSLogTransform, ScatterFilter
from clustering.case_collection import CaseCollection


PLOTDIR = "plots_test_som"

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


def create_cluster_pipe(min_samples=150):
    return Pipeline(steps=[
        # ("scale", StandardScaler()),
        ("clust", cluster.DBSCAN(eps=30, min_samples=min_samples, metric="euclidean")),
    ])


def plot_data(path, data, predictions, channels, title=None):
    fig = Figure(dpi=100, figsize=(8,4))
    FigureCanvas(fig)
    # autosize grid
    r = math.ceil(math.sqrt(len(channels)))
    overview, bcell = channels

    xchannel, ychannel = overview
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(data[xchannel], data[ychannel], s=1, c=predictions, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)


    xchannel, ychannel = bcell
    ax = fig.add_subplot(1, 2, 2)
    # subdata = data[predictions == 1]
    subdata = data
    ax.scatter(subdata[xchannel], subdata[ychannel], s=1, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)


def get_node_position(data, res, channels, position, max_val=1023, min_val=0):
    (c1, c2) = channels
    (p1, p2) = position
    x2 = max_val if p1 == "+" else min_val
    y2 = max_val if p2 == "+" else min_val
    closest = None
    cdist = None
    for i in np.unique(res):
        if i == -1:
            continue
        means = data.loc[res == i, channels].mean(axis=0)
        dist = math.sqrt((x2-means[c1])**2+(y2-means[c2])**2)
        if closest is None or cdist > dist:
            closest = i
            cdist = dist

    merged = [closest]
    for i in np.unique(res):
        if i == -1 or i == closest:
            continue
        means = data.loc[res == i, channels].mean(axis=0)
        dist = math.sqrt((x2-means[c1])**2+(y2-means[c2])**2)
        if abs(cdist - dist) < 250:
            print("Merging", i, "because dist diff", abs(cdist- dist))
            merged.append(i)

    print("Selected", closest)
    sel_res = np.zeros(res.shape)

    for m in merged:
        sel_res[res == m] = 1
    return sel_res

def process_fcs(filepath, label, plotdir, channels, pos, title=None):
    print("Processing", label)
    os.makedirs(plotdir, exist_ok=True)
    view_channels = list(combinations(channels, 2))

    meta, data = fcsparser.parse(filepath, encoding="latin-1")
    channel_info = print_meta(meta)


    scatter = ScatterFilter(
        [
            ("SS INT LIN", 0),
            ("FS INT LIN", 0),
            ("CD45-KrOr", 0),
            ("CD19-APCA750", 0),
        ]
    )
    print(data.shape)
    data = scatter.transform(data)
    print(data.shape)

    # sample_num = data.shape[0]
    # min_samples = 300 * (sample_num / 50000)
    # pipe = create_cluster_pipe(min_samples=min_samples)
    # res = pipe.fit_predict(data[channels].values)

    sel_res = get_node_position(data, res, channels, pos)

    view_channels = [
        ("CD45-KrOr", "SS INT LIN"),
        ("CD19-APCA750", "SS INT LIN"),
    ]
    plot_data(os.path.join(plotdir, label+"_kmeans"), data, sel_res, view_channels, title=title)
    sel_data = data[sel_res == 1]
    print(sel_data.shape)
    print("==========================")


def process_fcs_som(filepath, label, plotdir, channels, pos, title=None):
    os.makedirs(plotdir, exist_ok=True)
    view_channels = list(combinations(channels, 2))

    meta, data = fcsparser.parse(filepath, encoding="latin-1")
    channel_info = print_meta(meta)


    scatter = ScatterFilter(
        [
            ("SS INT LIN", 0),
            ("FS INT LIN", 0),
            ("CD45-KrOr", 0),
            ("CD19-APCA750", 0),
        ]
    )
    print(data.shape)
    data = scatter.transform(data)
    print(data.shape)

    pipe = create_pipeline()
    pipe.fit(data)

    weights = pipe.named_steps["clust"].model.output_weights
    df_weights = pd.DataFrame(weights)
    df_weights.columns = data.columns
    df_weights["predictions"] = df_weights.index

    # predictions = pipe.predict(data)
    # data["predictions"] = predictions

    # sample_num = data.shape[0]
    # min_samples = 300 * (sample_num / 50000)
    # pipe = create_cluster_pipe(min_samples=min_samples)
    # res = pipe.fit_predict(data[channels].values)

    # sel_res = get_node_position(data, res, channels, pos)

    view_channels = [
        ("CD45-KrOr", "SS INT LIN"),
        ("CD19-APCA750", "SS INT LIN"),
    ]
    plot_data(os.path.join(plotdir, label+"_som"), df_weights, df_weights["predictions"], view_channels, title=title)
    print("==========================")


    # plot_data(os.path.join(plotdir, label), data, view_channels)

    # plot_data(os.path.join(plotdir, label+"_nodes"), df_weights, view_channels)


# coll = CaseCollection("case_info.json", "mll-flowdata", tmpdir="tests/tube2")

# for label, group, fcsdata in coll.get_all_data(num=1, tube=2):
#     print(label, group)

sel_chans = ["CD45-KrOr", "SS INT LIN"]
position = ["+", "-"]

indir = "tests"
for cohort in os.listdir(indir):
    for i, filename in enumerate(os.listdir(os.path.join(indir, cohort))):
        filepath = os.path.join(indir, cohort, filename)
        process_fcs_som(filepath, "{}_{}".format(cohort, i), PLOTDIR, sel_chans, position, title=filename)
        break
    break

# indir = "tests"
# for cohort in os.listdir(indir):
#     for i, filename in enumerate(os.listdir(os.path.join(indir, cohort))):
#         filepath = os.path.join(indir, cohort, filename)
#         process_fcs(filepath, "{}_{}".format(cohort, i), PLOTDIR, sel_chans, position, title=filename)

# filepath = "tests/normalvar/65b28b75d6ef99db838c9c982637289f6600b901-2 CLL 9F 02 N07 001.LMD"
# process_fcs(filepath, "low_dens","single_tests", sel_chans, position)
# 

# filepath = "tests/over90/0851aee4afc582ffdf2683a38f7a05071bdebd66-3 CLL 9F 02 N10 001.LMD"
# process_fcs(filepath, "hi_dens_false","single_tests", sel_chans, position)

# filepath = "tests/normalvar/1b5ad3f1b9f9038e347223a61cf98470ec4318d8-3 CLL 9F 02 N11 001.LMD"
# process_fcs(filepath, "split_pop","single_tests", sel_chans, position)

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
