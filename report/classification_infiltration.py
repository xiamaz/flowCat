#!/usr/bin/env python3
from multiprocessing import pool
import pandas as pd

from report import prediction, file_utils
from report.stats import accuracy
from report.base import plot_figure


SELECTED_EXPERIMENTS = {
    "sqrt Transformed": ("abstract_single_groups_sqrt", "somgated", "random"),
}


def infil_process(indata):
    _, sel_row = indata
    pdata = file_utils.load_predictions(sel_row["predictions"])

    accs = []
    for infil in [0, 2.5, 5, 7.5, 10]:
        infil_data = {
            k: v.loc[(v["infiltration"] > infil) | (v["group"] == "normal")]
            for k, v in pdata.items()
        }
        sel_accs = accuracy.acc_table_dfs(infil_data)
        sel_accs["thres"] = infil
        accs.append(sel_accs)

    acc_df = pd.concat(accs)

    return acc_df


def process_infiltration(pred, exp_sel):
    """Create plots and tables for infiltration information on a single
    experiment."""

    exp_df = pred.classification.loc[exp_sel]

    with pool.Pool(12) as cpool:
        results = cpool.map(infil_process, exp_df.iterrows())

    avg_accs = pd.concat(
        results
    ).set_index(["thres"], append=True).mean(level=[1, 2])

    print(avg_accs)

    with plot_figure("infil_acc_sqrt") as axes:
        for name, data in avg_accs.groupby(level=0):
            axes.plot(
                data.index.get_level_values("thres"),
                data["micro"],
                label=name
            )

        axes.legend()


def main():
    pred = prediction.Prediction()
    for name, exp in SELECTED_EXPERIMENTS.items():
        print("Processing {}".format(name))
        process_infiltration(pred, exp)


if __name__ == "__main__":
    main()
