from sklearn import metrics
from flowcat import io_functions


def calculate_roc_curve(trues, preds, groups):
    curves = {}
    for i, group in enumerate(groups):
        curves[group] = metrics.roc_curve(trues[:, i], preds[:, i])
    return curves


def calculate_roc(trues, preds, groups):
    auc = {}
    for i, group in enumerate(groups):
        auc[group] = metrics.roc_auc_score(trues[:, i], preds[:, i])

    macro_auc = metrics.roc_auc_score(trues, preds, average="macro")
    micro_auc = metrics.roc_auc_score(trues, preds, average="micro")
    results = {
        "one-vs-rest": auc,
        "macro": macro_auc,
        "micro": micro_auc,
    }
    return results


def plot_roc_curves(ax, curves):
    """Plot given curves to the ax element."""
    for name, curve in curves.items():
        ax.plot(curve[0], curve[1], label=name)

    ax.plot((0, 1), (0, 1), "k--")
    ax.legend()
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC one-vs-rest")
