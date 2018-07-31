"""Extended confusion matrix analysis. Including hierarchical clustering of
results."""
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from .base import Reporter


def calc_micro_acc(confusion):
    """Calculate micro accuracy by dividing the diagonal with the
    overall sum."""
    shape = confusion.shape
    assert len(shape) == 2 and shape[0] == shape[1], "Not n x n matrix."

    diagonal_sum = sum(confusion.iloc[i, i] for i in range(shape[0]))
    total_sum = np.sum(confusion.values)
    accuracy = diagonal_sum / total_sum
    return accuracy


def calc_macro_acc(confusion):
    """Calculate correct classifications for each group."""
    shape = confusion.shape
    assert len(shape) == 2 and shape[0] == shape[1], "Not n x n matrix."

    row_sums = np.sum(confusion, axis=1)
    group_accuracies = [
        confusion.iloc[i, i] / row_sums[i]
        for i in range(shape[0])
    ]
    return np.mean(group_accuracies)


def normalize_confusion(confusion):
    """Normalize results in the confusion matrx."""
    return confusion / confusion.sum(axis=1)


def calc_hierarchical(confusion, path):
    """Calculate hierarchical clustering for the given confusion matrix."""

    distance = pdist(confusion, metric="euclidean")
    linkage = hierarchy.linkage(distance, method="single")
    plt.figure()
    hierarchy.dendrogram(
        linkage, show_contracted=True, labels=confusion.index.tolist()
    )
    plt.savefig(path)
    plt.close()


class Confusion(Reporter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_confusion_matrices(self, exp_set):
        metadata = exp_set.apply(self.get_metadata, axis=1)

        # single experiment in experiment set
        matrices = []
        for single_exp in metadata:
            # one experiment using a different base SOM dataset
            confusions = []
            groups_list = []
            for subexp in single_exp:
                # sum up all different results from the different kfold splits
                summed_confusion = reduce(lambda a, b: a + b, [
                    np.matrix(c["confusion"])
                    # different configs can be run in a single run
                    for c in subexp["experiments"]
                ])
                confusions.append(summed_confusion)
                groups_list.append(subexp["experiments"][0]["groups"])

            groups = groups_list[0]
            # assert that the ordering is the same
            for group in groups_list[1:]:
                assert all([group[i] == groups[i] for i in range(len(group))])
            # avg_confusion = pd.DataFrame(
            #     data=np.mean(confusions, axis=0),
            #     columns=groups,
            #     index=groups,
            # )
            matrices.append([
                pd.DataFrame(data=c, columns=groups, index=groups)
                for c in confusions
            ])
            # var_confusion = np.var(confusions, axis=0)

            # sum statistics in a way that takes deviation into account

        # calculate summed up results but also show individual results
        exp_set.loc[:, "confusion"] = matrices
        return exp_set

    def calc_confusion_matrix(self, exp_row):
        """Provide higher accuracy confusion matrix numbers."""
        # unique identifying name
        name = "{}: {} {}".format(
            exp_row["set"], exp_row["name"], exp_row["type"]
        )

        confusions = exp_row["confusion"]

        micro_accs = list(map(calc_micro_acc, confusions))
        micro_acc = np.mean(micro_accs)
        micro_std = np.std(micro_accs)
        macro_accs = list(map(calc_macro_acc, confusions))
        macro_acc = np.mean(macro_accs)
        macro_std = np.std(macro_accs)

        print(name)
        print("Micro Average Accuracy: {} +- {}".format(micro_acc, micro_std))
        print("Macro Average Accuracy: {} +- {}".format(macro_acc, macro_std))

        return micro_accs

    def plot_confusion_matrix(self, exp_row):
        """Plot dendrogram from provided confusion matrix."""
        # unique identifying name
        name = "{}: {} {}".format(
            exp_row["set"], exp_row["name"], exp_row["type"]
        )
        confusion = exp_row["confusion"]
        calc_hierarchical(confusion)
