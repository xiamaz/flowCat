import numpy as np
import pandas as pd

from matplotlib import cm

import sys
sys.path.append("../classification")
from classification import plotting


metadata = pd.read_csv("./sommaps/huge_s30.csv")
group_map = {
    "CLL": "CM",
    "MBL": "CM",
    "MZL": "LM",
    "LPL": "LM",
    "MCL": "MP",
    "PL": "MP",
}
metadata["group"] = metadata["group"].apply(lambda g: group_map.get(g, g))

groups = list(metadata["group"].unique())

weights = np.ones((len(groups), len(groups)), dtype=int)
plotting.plot_confusion_matrix(
    weights, classes=groups, normalize=False, filename="weights_normal.png", dendroname=None, cmap=cm.Reds
)
for i in range(len(groups)):
    if i != groups.index("normal"):
        weights[i][groups.index("normal")] = 50
        weights[groups.index("normal")][i] = 25


plotting.plot_confusion_matrix(
    weights, classes=groups, normalize=False, filename="weights_biased.png", dendroname=None, cmap=cm.Reds
)
