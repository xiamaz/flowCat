#!/usr/bin/env python3
"""
Validate generated soms by creating a tsne plot using a random stratified
subsample of test data."""

import numpy as np
from sklearn import manifold

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

from argmagic import argmagic
from flowcat import utils, io_functions, mappings


def create_tsne(data: utils.URLPath, meta: utils.URLPath, plotdir: utils.URLPath):
    """Generate tsne plots for a subsample of data.

    Args:
        data: Path to generated soms for cases.
        meta: Path to metadata json for cases.
        plotdir: Path to output plots for data.
    """
    # data = flowcat.utils.URLPath("output/test-2019-08/som")
    # meta = flowcat.utils.URLPath("output/test-2019-08/som.json")
    # plotdir = flowcat.utils.URLPath("output/test-2019-08/tsne")

    cases = io_functions.load_case_collection(data, meta)

    # cases = cases.sample(20, flowcat.mappings.GROUPS)
    # flowcat.io_functions.save_json(cases.labels, plotdir / "case_ids.json")
    # labels = io_functions.load_json(plotdir / "case_ids.json")
    # cases = cases.filter(labels=labels)
    # print(cases)

    groups = np.array([case.group for case in cases])

    colors = {
        "CLL": "red",
        "MBL": "dodgerblue",
        "MCL": "steelblue",
        "PL": "skyblue",
        "LPL": "limegreen",
        "MZL": "forestgreen",
        "FL": "springgreen",
        "HCL": "orchid",
        "normal": "darkgoldenrod",
    }

    plotdir.mkdir()

    for tube in ("1", "2", "3"):
        soms = []
        for case in cases:
            sample = case.get_tube(tube, kind="som")
            sample.path = data / f"{case.id}_t{tube}.npy"
            som = sample.get_data().data.flatten()
            soms.append(som)
        somdata = np.array(soms)

        tsne = manifold.TSNE(n_components=2, perplexity=10)
        transformed = tsne.fit_transform(somdata)

        fig, ax = plt.subplots(figsize=(11, 7))
        for group in mappings.GROUPS:
            gdata = transformed[groups == group]
            gx = gdata[:, 0]
            gy = gdata[:, 1]
            ax.scatter(gx, gy, c=colors[group], label=group)

        plt.legend()
        plt.savefig(plotdir / f"tsne_{tube}.png")
        plt.close("all")


if __name__ == "__main__":
    argmagic(create_tsne)
