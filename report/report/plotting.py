from functools import reduce
from contextlib import contextmanager

import pandas as pd

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import cm
from matplotlib.figure import Figure

# import altair as alt

from .prediction import df_stats, roc, auc_one_vs_all


def avg_stats_plot(somiter_data: dict) -> "alt.Chart":
    pass
#     """Average df results from multiple dataframes."""
#     som_df = pd.concat(
#         [df_stats(d) for d in somiter_data.values()], keys=somiter_data.keys()
#     )
#     som_df["incorrect"] = 1.0 - (
#         som_df["correct"] + som_df["uncertain"]
#     )
#     mean = som_df.mean(level=[1, 2])
#     # std = som_df.std(level=[1, 2])
# 
#     alt_df = mean.stack().reset_index()
#     alt_df.columns = ["cohort", "stat", "type", "val"]
#     chart = alt.Chart(alt_df).mark_bar().encode(
#         x=alt.X("stat:N", axis=alt.Axis(title="")),
#         y=alt.Y("sum(val):Q", axis=alt.Axis(title="", grid=False)),
#         column=alt.Column("cohort:N"),
#         color=alt.Color(
#             "type:N",
#             sort=["incorrect", "uncertain", "correct"],
#             scale=alt.Scale(range=["#ff7e7e", "#FFB87E", "#77F277"]),
#         ),
#         order="ttype:N",
#     ).transform_calculate(
#         ttype=(
#             "if(datum.type == 'uncertain', 1, "
#             "if(datum.type == 'correct', 0, 2))"
#         )
#     )
#     return chart


def roc_plot(roc_data: dict, auc: dict, ax: "Axes") -> Figure:
    colors = cm.tab20.colors  # pylint: disable=no-member
    for i, (name, data) in enumerate(roc_data.groupby(level=0)):
        ax.plot(
            data["fpr"],
            data["tpr"],
            color=colors[i * 2], lw=1,
            label="{} (AUC {:.2f})".format(name, auc[name])
        )
        ax.fill_between(
            data["fpr"],
            data["tpr"] - data["std"],
            data["tpr"] + data["std"],
            color=colors[i * 2 + 1],
            alpha=1 - (1 / len(auc) * i),
        )

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_ylim((0, 1.05))
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim((0, 1.0))
    ax.set_xlabel("False Positive Rate")
    ax.set_title("One vs all")
    ax.legend()
    return ax


def avg_roc_plot(somiter_data: dict, ax: "Axes") -> "Axes":
    """Create a figure containing average auc data."""
    roc_curves = pd.concat(
        [roc(d) for d in somiter_data.values()], keys=somiter_data.keys()
    )
    roc_curves.index.rename("som", level=0, inplace=True)

    bin_index = pd.interval_range(start=0, end=1, periods=100)
    roc_curves["bin"] = pd.cut(roc_curves["fpr"], bin_index)
    roc_curves.set_index("bin", append=True, inplace=True)
    roc_mean = roc_curves.mean(
        level=["positive", "bin"]
    ).interpolate()
    roc_mean["std"] = roc_curves["tpr"].std(
        level=["positive", "bin"]
    ).interpolate()
    auc_stats = {
        k: v / len(somiter_data) for k, v in reduce(
            lambda x, y: {k: v + y[k] for k, v in x.items()},
            [auc_one_vs_all(d) for d in somiter_data.values()]
        ).items()
    }
    ax = roc_plot(roc_mean, auc_stats, ax=ax)
    return ax


def plot_avg_roc_curves(somiter_data: dict) -> Figure:
    """Plot average ROC curves for the given data."""
    fig = Figure()

    ax = fig.add_subplot(111)
    avg_roc_plot(somiter_data, ax)
    fig.set_size_inches(8, 8)
    fig.tight_layout(rect=[0, 0, 1, 1])
    FigureCanvas(fig)
    return fig


def experiments_plot(data: pd.DataFrame) -> "alt.Chart":
    pass
#     """Plot information on experiments into a bar chart.
#     Chart has to be multiindexed into set, name and type.
#     """
#     data.reset_index(inplace=True)
# 
#     charts = []
#     for sname, sdata in data.groupby("set"):
#         parts = []
#         for ename, edata in sdata.groupby("name"):
#             base = alt.Chart(edata).mark_bar().encode(
#                 y=alt.Y("type:N", axis=alt.Axis(title=ename)),
#                 x=alt.X(
#                     "count:Q",
#                     axis=alt.Axis(title=""),
#                     scale=alt.Scale(domain=(0, 10))
#                 ),
#                 color="type:N",
#             )
#             parts.append(base)
#         part = alt.vconcat(*parts)
#         part.title = sname
#         charts.append(part)
# 
#     chart = alt.hconcat(*charts).configure(
#         axis=alt.AxisConfig(
#             titleAngle=0,
#             titleLimit=0,
#             titleAlign="left",
#             titleX=0,
#             titleY=0,
#         ),
#         title=alt.VgTitleConfig(
#             offset=20
#         )
#     )
#     return chart


def plot_frequency(data: pd.DataFrame, path: str):
    pass
#     """Set frequency of cases against certainty with standard deviation"""
# 
#     plt_data = data.reset_index()
#     plt_data.sort_values("macro", inplace=True)
# 
#     chart = alt.Chart(plt_data).mark_point().encode(
#         x="macro",
#         y="mean",
#         color="group",
#     )
#     chart.save(path)
