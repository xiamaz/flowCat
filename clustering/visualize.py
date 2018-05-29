import math
import logging
from itertools import combinations
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn import cluster
from sklearn.pipeline import Pipeline

import fcsparser

from clustering.clustering import (
    create_pipeline, ScatterFilter, GatingFilter, SOMGatingFilter
)
from clustering.case_collection import CaseCollection
from clustering.plotting import plot_overview


PLOTDIR = "plots_test_som"

logger = logging.getLogger("clustering")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

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


def create_cluster_pipe(min_samples=150, eps=30):
    return Pipeline(steps=[
        # ("scale", StandardScaler()),
        ("clust", cluster.DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")),
    ])


def plot_data(path, data, predictions, channels, somweights, somfirst, somfirst_clusters, somfirst_predictions, title=None, seq=True):
    fig = Figure(dpi=100, figsize=(12,12)) # width, height
    FigureCanvas(fig)
    # autosize grid
    r = math.ceil(math.sqrt(len(channels)))
    overview, bcell = channels

    xchannel, ychannel = overview
    ax = fig.add_subplot(3, 3, 1)
    ax.scatter(data[xchannel], data[ychannel], s=1, c=predictions, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)
    ax.set_title("DBSCAN cluster search")

    xchannel, ychannel = bcell
    ax = fig.add_subplot(3, 3, 2)
    if seq:
        subdata = data[predictions == 1]
    else:
        subdata = data
    ax.scatter(subdata[xchannel], subdata[ychannel], s=1, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)
    ax.set_title("Gated CD19")

    xchannel, ychannel = bcell
    gated = True
    ax = fig.add_subplot(3, 3, 5)
    ax.scatter(somweights[gated][xchannel], somweights[gated][ychannel], s=1, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)
    ax.set_title("Gated CD19 SOM")

    xchannel, ychannel = bcell
    ax = fig.add_subplot(3, 3, 3)
    ax.scatter(data[xchannel], data[ychannel], s=1, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)
    ax.set_title("Ungated CD19")

    xchannel, ychannel = bcell
    gated = False
    ax = fig.add_subplot(3, 3, 6)
    ax.scatter(somweights[gated][xchannel], somweights[gated][ychannel], s=1, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)
    ax.set_title("Ungated CD19 SOM")

    xchannel, ychannel = overview
    ax = fig.add_subplot(3, 3, 7)
    ax.scatter(somfirst[xchannel], somfirst[ychannel], s=4, c=somfirst_clusters, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)
    ax.set_title("SOM first DBSCAN")

    xchannel, ychannel = bcell
    somgated = data[somfirst_predictions == 1]
    ax = fig.add_subplot(3, 3, 8)
    ax.scatter(somgated[xchannel], somgated[ychannel], s=4, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)
    ax.set_title("SOM first Gated CD19")

    xchannel, ychannel = bcell
    ax = fig.add_subplot(3, 3, 9)
    ax.scatter(somfirst[xchannel], somfirst[ychannel], s=4, c=somfirst_clusters, marker=".")
    ax.set_xlabel(xchannel)
    ax.set_ylabel(ychannel)
    ax.set_title("SOM first Ungated CD19")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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

    sample_num = data.shape[0]
    min_samples = 300 * (sample_num / 50000)
    pre_pipe = create_cluster_pipe(min_samples=min_samples)
    res = pre_pipe.fit_predict(data[channels].values)

    sel_res = get_node_position(data, res, channels, pos)

    sel_data = data[sel_res == 1]
    pipe = create_pipeline()
    pipe.fit(sel_data)
    weights = pipe.named_steps["clust"].model.output_weights
    gated_df_weights = pd.DataFrame(weights)
    gated_df_weights.columns = sel_data.columns
    gated_df_weights["predictions"] = gated_df_weights.index

    pipe = create_pipeline()
    pipe.fit(data)
    weights = pipe.named_steps["clust"].model.output_weights
    df_weights = pd.DataFrame(weights)
    df_weights.columns = data.columns
    df_weights["predictions"] = df_weights.index

    somweights = {
        True: gated_df_weights,
        False: df_weights,
    }

    # somfirst predictions
    eps = 30
    min_samples = 4
    somfirst_pipe = create_cluster_pipe(min_samples=min_samples, eps=eps)
    somfirst_result = somfirst_pipe.fit_predict(df_weights[channels].values)
    som_node_predictions = get_node_position(df_weights, somfirst_result, channels, pos)

    data_predictions = pipe.predict(data)

    somfirst_predictions = np.vectorize(lambda x: som_node_predictions[x])(data_predictions)
    print(somfirst_predictions)

    view_channels = [
        ("CD45-KrOr", "SS INT LIN"),
        ("CD19-APCA750", "SS INT LIN"),
    ]
    plot_data(os.path.join(plotdir, label+"_kmeans"), data, sel_res, view_channels, somweights, df_weights, somfirst_result, somfirst_predictions, title=title, seq=True)

    print("==========================")


def process_consensus(data, plotdir, channels, positions, tube):
    os.makedirs(plotdir, exist_ok=True)

    pipe = Pipeline(steps=[
        ("scatter", ScatterFilter()),
        ("somgating", SOMGatingFilter(channels, positions)),
    ])

    for i, d in enumerate(data):
        print(d.shape)

        t = pipe.fit_transform(d)

        subplotdir = os.path.join(plotdir, "single")
        os.makedirs(subplotdir, exist_ok=True)

        plotpath = os.path.join(subplotdir, "{}_all".format(i))
        fig = Figure()
        gs = GridSpec(3, 1)

        rows = 0
        aw = 0
        ah = 0
        fig, (nw, nh) = plot_overview(d, tube, fig=fig, gs=gs, gspos=rows, title="raw ungated data")
        rows += 1
        aw = max(aw, nw)
        ah += nh
        somgating = pipe.named_steps["somgating"]
        fig, (nw, nh) = plot_overview(
            somgating.som_weights, coloring=somgating.som_to_clust, s=8,
            fig=fig, gs=gs, gspos=rows, title="clustering soms"
        )
        rows += 1
        aw = max(aw, nw)
        ah += nh

        fig, (nw, nh) = plot_overview(t, tube, fig=fig, gs=gs, gspos=rows, title="som gated data")
        rows += 1
        aw = max(aw, nw)
        ah += nh
        fig.set_size_inches(aw, ah)
        FigureCanvas(fig)
        gs.tight_layout(fig)
        # fig.tight_layout()
        fig.savefig(plotpath, dpi=100)


CHANNELS = ["CD45-KrOr", "SS INT LIN"]
POSITIONS = ["+", "-"]
TUBE = 1

# PLOTDIR = "plots_test_seq"
# indir = "tests"
# for cohort in os.listdir(indir):
#     for i, filename in enumerate(os.listdir(os.path.join(indir, cohort))):
#         filepath = os.path.join(indir, cohort, filename)
#         process_fcs(filepath, "{}_{}".format(cohort, i), PLOTDIR, sel_chans, position, title=filename)
#         break

with open("selected_hashed_ids.json") as selfile:
    refcases = [c for cc in json.load(selfile).values() for c in cc]

collection = CaseCollection("case_info.json", "mll-flowdata", "tests_consensus")

concat_files = collection.get_train_data(labels=refcases, num=None, groups=None, tube=TUBE)

process_consensus(concat_files, "plots_consensus", CHANNELS, POSITIONS, TUBE)
