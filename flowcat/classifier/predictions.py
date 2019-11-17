import pandas as pd
from sklearn import metrics
from flowcat import io_functions
from flowcat.plots import confusion as plot_confusion


def map_labels(labels, mapping):
    """Map labels to new labels defined by mapping."""
    return [mapping.get(l, l) for l in labels]


def generate_confusion(true_labels, pred_labels, groups, output):
    """Calculate confusion matrix metrics and also create plots."""
    confusion = metrics.confusion_matrix(true_labels, pred_labels, labels=groups)
    confusion = pd.DataFrame(confusion, index=groups, columns=groups)
    print(confusion)
    io_functions.save_csv(confusion, output / "validation_confusion.csv")

    plot_confusion.plot_confusion_matrix(
        confusion, normalize=False).savefig(str(output / "confusion_abs.png"), dpi=300)
    plot_confusion.plot_confusion_matrix(
        confusion, normalize=True).savefig(str(output / "confusion_norm.png"), dpi=300)
    return confusion


def generate_metrics(true_labels, pred_labels, groups, output):
    """Generate numeric metrics."""
    metrics_results = {
        "balanced": metrics.balanced_accuracy_score(true_labels, pred_labels),
        "f1_micro": metrics.f1_score(true_labels, pred_labels, average="micro"),
        "f1_macro": metrics.f1_score(true_labels, pred_labels, average="macro"),
        "mcc": metrics.matthews_corrcoef(true_labels, pred_labels),
    }
    print(metrics_results)
    io_functions.save_json(metrics_results, output / "validation_metrics.json")
    return metrics_results


def generate_all_metrics(true_labels, pred_labels, mapping, output):
    """Create metrics and confusion matrix and save them to the given output directory.
    """
    output.mkdir()

    groups = mapping["groups"]
    map_dict = mapping["map"]
    if map_dict:
        true_labels = map_labels(true_labels, map_dict)
        pred_labels = map_labels(pred_labels, map_dict)

    confusion = generate_confusion(true_labels, pred_labels, groups, output)
    metrics = generate_metrics(true_labels, pred_labels, groups, output)

    return confusion, metrics
