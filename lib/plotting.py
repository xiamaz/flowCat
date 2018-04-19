'''
Learning visualization functions
'''
import os

import itertools
import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn
# necessary for headless usage


def plot_confusion_matrix(confusion_matrix: "numpy.matrix", classes: [str],
                          normalize: bool = False,
                          title: str = 'Confusion matrix',
                          cmap=plt.cm.Blues, filename: str = 'confusion.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / \
            confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.close('all')
    plt.figure()

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]),
                                  range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def plot_splits(splits, path):
    '''Plot splits with group label distributions.'''
    df_data = [
        {
            "i": i,
            "size": size,
            "group": name
        }
        for i, s in enumerate(splits)
        for j, (name, size) in enumerate(s["groups"].items())
    ]
    df_data = pandas.DataFrame(df_data)

    grouped = seaborn.factorplot(
        x="i", hue="group", y="size", data=df_data,
        size=6, kind="bar", palette="muted"
    )

    grouped.despine(left=True)
    grouped.set_ylabels("Group size.")
    grouped.set_titles("Split group sizes.")

    grouped.savefig(path)


def plot_single(results: list, path: str, name: str) -> None:
    '''Plot runs from single experiment.'''
    for result in results:
        split_path = os.path.join(
            path, "000-{}_{}_split_groups.png".format(
                name, result["setting"]
            )
        )
        plot_splits(result["splits"], split_path)


def plot_change(data, target, path, xlabel="Size change"):
    '''Plot change in data according to specified target.'''

    output = os.path.join(path, "{}_combined.png".format(target))
    plt.close('all')

    axes = data.plot(x="set", y=target)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(target)
    fig = axes.get_figure()

    fig.suptitle("{} by size".format(target))

    plt.tight_layout()
    fig.savefig(output, dpi=300)

    plt.close(fig)


def plot_combined(results: list, path: str) -> None:
    '''Plot results to path directory.'''
    output_path = os.path.join(path, "overview_plots")
    os.makedirs(output_path, exist_ok=True)

    # analyze splits
    avg_stats = []
    for i, result in results:
        plot_single(result, output_path, str(i))
        avg_stats += [
            dict({
                "setting": r["setting"],
                "set": i
            }, **{
                k: v for k, v in r["avg_result"].items()
            })
            for r in result
        ]

    avg_stats = pandas.DataFrame(avg_stats)

    plot_change(avg_stats, "accuracy", output_path)
    plot_change(avg_stats, "f1", output_path)
    plot_change(avg_stats, "precision", output_path)
