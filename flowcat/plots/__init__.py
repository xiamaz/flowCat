import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from flowcat import io_functions
from . import roc as fc_roc


def create_roc_results(trues, preds, model, output):
    """Create ROC and AUC metrics and save them to the given directory."""
    groups = model.config.groups
    roc = fc_roc.calculate_roc(trues, preds, groups)
    io_functions.save_json(roc, output / "auc.json")

    fig, ax = plt.subplots()
    fc_roc.plot_roc_curves(ax, fc_roc.calculate_roc_curve(trues, preds, groups))

    fig.tight_layout()
    fig.savefig(str(output / "roc.png"), dpi=300)
