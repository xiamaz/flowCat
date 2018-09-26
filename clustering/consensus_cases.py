"""
Get a list of cases with which are misclassified in a lot of different runs.
"""
import collections
import numpy as np
import pandas as pd

used_runs = [
    "mll-sommaps/models/smallernet_double_yesglobal_epochrand_sommap_8class/predictions_0.csv",
    "mll-sommaps/models/smallernet_double_yesglobal_sommap_8class/predictions_0.csv",
    "mll-sommaps/models/convolutional_2x2filter_yesregu_epochrand_sommap_8class/predictions_0.csv",
    "mll-sommaps/models/deepershift_counts_sommap_8class/predictions_0.csv",
    "mll-sommaps/models/deepershift_counts_noweight_sommap_8class/predictions_0.csv",
    "mll-sommaps/models/deepershift_counts_noweight_moreregu_sloweradam_sommap_8class/predictions_0.csv",
]

def load_data(path):
    return pd.read_csv(path, index_col=0)


def add_correct_magnitude(data):
    newdata = data.copy()
    valcols = [c for c in data.columns if c != "correct"]
    selval = np.vectorize(lambda i: valcols[i])
    newdata["largest"] = data[valcols].max(axis=1)
    newdata["pred"] = selval(data[valcols].values.argmax(axis=1))
    return newdata


def main():
    used_frames = pd.concat(
        map(add_correct_magnitude, map(load_data, used_runs)),
        levels=range(len(used_runs)))

    amount_all = collections.defaultdict(int)
    amount_maj = collections.defaultdict(int)
    maj_labels = []
    for gname, gdata in used_frames.groupby(level=0):
        misclassif = gdata["pred"] != gdata["correct"]
        perc = sum(misclassif) / len(misclassif)
        orig_group = gdata["correct"].iloc[0]
        if perc > 0.5:
            falseclas = list(gdata.loc[misclassif, "pred"])
            amount_maj[orig_group] += 1
            maj_labels.append(
                {"label": gname, "group": orig_group, "perc": perc, "misclassif": falseclas}
            )
            if perc == 1.0:
                amount_all[orig_group] += 1

    maj_df = pd.DataFrame(maj_labels)
    maj_df.to_csv("misclassified.csv")


if __name__ == "__main__":
    main()
