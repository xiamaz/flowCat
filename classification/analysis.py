import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

path = "output/classification/initial_comp_indiv_pregating_dedup_20180613_1632"

csvs = [
    os.path.join(path, d, f)
    for d in os.listdir(path)
    for f in os.listdir(os.path.join(path, d)) if f.endswith("predictions.csv")
]

def p(*a):
    print(*a)
    return a

# just test on the first one
path_to_df = lambda p: pd.read_table(p, delimiter=",", index_col=0)
pred_diag = lambda r: r[[isinstance(v, float) for v in r]].drop("infiltration").astype("float32").idxmax()
correct_diag = lambda r: pred_diag(r) != r["group"]
filt_df = lambda d: d[d.apply(correct_diag, axis=1)]

seq = lambda p: filt_df(path_to_df(p))

dfs = map(seq, csvs)

all_mis = pd.concat(list(dfs))

con = Counter(list(all_mis.index))

plt.figure()
plt.hist(list(con.values()), bins=len(csvs), range=(1, len(csvs)))
plt.title("Misclassifications in pregated results")
plt.savefig("MIS_HIST_pregated_dedup_merged_classes.png")
plt.close()

over = {k: v for k, v in con.items() if v == 9}
len(over)

max_misses = all_mis.loc[all_mis.index.isin(list(over.keys()))]
label_groups = list(max_misses.groupby(max_misses.index))
res = []
for l, d in label_groups:
    diags = Counter(d.apply(pred_diag, axis=1))
    res.append((l, set(d["group"]), diags))


groups = []
for r in res:
    print(r)
    _, g, _ = r
    groups.append(list(g)[0])


plt.figure()
plt.hist(groups)
plt.title("Groups in highest misclassified in pregated results")
plt.savefig("groups_highest_pregated.png")
plt.close()


def sel_corr(r):
    return r[r["group"]]

all_data = pd.concat(map(path_to_df, csvs))
all_data["corr"] = all_data.apply(sel_corr, axis=1)
for gname, data in all_data.groupby("group"):
    plt.figure()
    plt.title("Infiltration v Certainty in {}".format(gname))
    plt.scatter(data["corr"], data["infiltration"], s=1)
    plt.xlabel("Certainty in classification of correct label.")
    plt.ylabel("Infiltration rate")
    plt.savefig("scatter_infil_{}.png".format(gname))
    plt.close()

all_data["pred"] = all_data.apply(pred_diag, axis=1)
for gname in all_data["group"].unique():
    print(gname)
    plt.figure()
    plt.title("Classification certainty for {}".format(gname))
    gsorted = all_data.sort_values(gname)
    colors = np.array(['b']*gsorted.shape[0])
    colors[gsorted["pred"]==gname] = "r"
    plt.bar(list(range(gsorted.shape[0])), gsorted[gname], width=1.0, color=colors)
    plt.savefig("certainty_{}.png".format(gname))
    plt.close()



xts = PCA(n_components=2).fit_transform(all_data.drop(["group", "infiltration", "corr", "pred"], axis=1))

colors = [plt.cm.jet(float(i)/4) for i in range(4)]

plt.figure()
plt.title("tSNE of classification results")
for i, name in enumerate(all_data["group"].unique()):
    d = xts[all_data["group"]==name]
    plt.scatter(d[:, 0], d[:, 1], c=colors[i], label=name)
plt.legend()
plt.savefig("pca_predictions.png")
plt.close()



overviewsp = {p:os.path.join("output/classification", p, "overview_plots/avg_stats.csv") for p in os.listdir("output/classification") if "dedup" in p}

overviews = {
    k.replace("initial_comp_", "").split("_dedup_")[0]: pd.read_table(v, sep=",", index_col=0)
    for k, v in overviewsp.items()
}

over_df = pd.DataFrame()
over_df["name"] = list(overviews.keys())
over_df["f1"] = [d.loc["mean", "f1"] for d in overviews.values()]
over_df["std"] = [d.loc["std", "f1"] for d in overviews.values()]

over_df["groups"] = "merged"
over_df.loc[over_df["name"].apply(lambda x: "all_groups" in x), "groups"] = "all"
over_df["name"] = over_df["name"].apply(lambda x: x.replace("all_groups_", ""))

over_df.set_index(["groups", "name"], inplace=True)
