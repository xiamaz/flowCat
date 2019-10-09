"""
Test the performance and behavior on samples from unknown cohorts, such as
AML, MM and HCLv. (thus covering: outside B-NHL and unknown B-NHL subtypes).
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")
from sklearn import metrics
import keras
from argmagic import argmagic

from flowcat import io_functions, utils, som_dataset


class SOMClassifier:
    def __init__(self, model, binarizer, config, data_ids: dict = None):
        self.model = model
        self.config = config
        self.binarizer = binarizer
        self.data_ids = data_ids

    @classmethod
    def load(cls, path: utils.URLPath):
        """Load classifier model from the given path."""
        config = io_functions.load_json(path / "config.json")
        model = keras.models.load_model(str(path / "model.h5"))
        binarizer = io_functions.load_joblib(path / "binarizer.joblib")

        data_ids = {
            "validation": io_functions.load_json(path / "ids_validate.json"),
            "train": io_functions.load_json(path / "ids_train.json"),
        }
        return cls(model, binarizer, config, data_ids=data_ids)

    def get_validation_data(self, dataset: som_dataset.SOMDataset) -> som_dataset.SOMDataset:
        return dataset.filter(labels=self.data_ids["validation"])

    def create_sequence(
        self,
        dataset: som_dataset.SOMDataset,
        batch_size: int = 128
    ) -> som_dataset.SOMSequence:

        def getter(data, tube):
            return data.get_tube(tube, kind="som").get_data().data

        seq = som_dataset.SOMSequence(
            dataset, self.binarizer,
            get_array_fun=getter,
            tube=self.config["tubes"],
            batch_size=batch_size,
            pad_width=self.config["pad_width"],
        )
        return seq


def create_roc_results(trues, preds, output, model):
    """Create ROC and AUC metrics and save them to the given directory."""
    output.mkdir()
    curves = {}
    for i, group in enumerate(model.config["groups"]):
        curves[group] = metrics.roc_curve(trues[:, i], preds[:, i])

    auc = {}
    for i, group in enumerate(model.config["groups"]):
        auc[group] = metrics.roc_auc_score(trues[:, i], preds[:, i])

    macro_auc = metrics.roc_auc_score(trues, preds, average="macro")
    micro_auc = metrics.roc_auc_score(trues, preds, average="micro")
    io_functions.save_json(
        {
            "one-vs-rest": auc,
            "macro": macro_auc,
            "micro": micro_auc,
        },
        output / "auc.json")

    fig, ax = plt.subplots()
    for name, curve in curves.items():
        ax.plot(curve[0], curve[1], label=name)

    ax.plot((0, 1), (0, 1), "k--")
    ax.legend()
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC one-vs-rest")

    fig.tight_layout()
    fig.savefig(str(output / "roc.png"), dpi=300)
    plt.close()


def create_threshold_results(trues, preds, output, model):
    """Create threshold results from true and predicted."""
    # calculate accuracy for a certain certainty
    # how about w score above 0.95?
    output.mkdir()
    threshold_results = []
    for threshold in np.arange(0.25, 1.0, 0.05):
        index_above = np.argwhere(np.any(preds > threshold, axis=1)).squeeze()
        sel_preds = preds[index_above, :]
        sel_trues = trues[index_above, :]
        pred_labels = model.binarizer.inverse_transform(sel_preds)
        true_labels = model.binarizer.inverse_transform(sel_trues)
        included = len(index_above) / len(preds)
        acc = metrics.accuracy_score(true_labels, pred_labels)
        print(threshold, included, acc)
        threshold_results.append((threshold, included, acc))
    io_functions.save_json(threshold_results, output / "thresholds.json")

    tarr = np.array(threshold_results)
    fig, ax = plt.subplots()
    ax.plot(tarr[:, 0], tarr[:, 1], label="included")
    ax.plot(tarr[:, 0], tarr[:, 2], label="acc")
    ax.legend()
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Classification accuracy / Included cases ratio")
    fig.savefig(str(output / "threshold.png"), dpi=300)


def plot_embedded(transformed, true_labels, groups, colors, title=""):
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, group in enumerate(groups):
        sel_dots = transformed[np.array([i == group for i in true_labels]), :]
        ax.scatter(sel_dots[:, 0], sel_dots[:, 1], label=group, s=16, marker="o", c=[colors[i]])
    ax.legend()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def main(data: utils.URLPath, model: utils.URLPath, output: utils.URLPath):
    data, model, output = map(utils.URLPath, (
        "output/som-fix-test/soms-test/som_r4_1",
        "output/0-final/classifier-minmax-new",
        "output/0-final/model-analysis"
    ))
    dataset = io_functions.load_case_collection(data, data + ".json")
    dataset.set_data_path(utils.URLPath(""))

    model = SOMClassifier.load(model)
    validate = model.get_validation_data(dataset)
    val_seq = model.create_sequence(validate)

    trues = np.concatenate([val_seq[i][1] for i in range(len(val_seq))])
    preds = np.array([p for p in model.model.predict_generator(val_seq)])

    create_roc_results(trues, preds, output / "roc", model)
    create_threshold_results(trues, preds, output / "threshold", model)

    # tsne of result vectors
    embedding_path = output / "embedding-preds"
    embedding_path.mkdir()

    pred_labels = val_seq.true_labels
    groups = model.config["groups"]
    groups.remove("normal")
    groups = ["normal", *groups]
    all_groups = groups + ["AML", "MM", "HCLv"]
    colors = sns.cubehelix_palette(len(all_groups), rot=4, dark=0.30)

    from sklearn import manifold
    from umap import UMAP
    perplexity = 50

    # tsne of intermediate layers
    intermediate_model = keras.Model(
        inputs=model.model.input, outputs=model.model.get_layer("concatenate_1").output)
    intermed_preds = np.array([p for p in intermediate_model.predict_generator(val_seq)])

    # unknown data
    udata = utils.URLPath("output/unknown-cohorts-processing/som/som")
    udataset = io_functions.load_case_collection(udata, udata + ".json")
    udataset.set_data_path(utils.URLPath(""))
    un_seq = model.create_sequence(udataset)
    intermed_upreds = np.array([p for p in intermediate_model.predict_generator(un_seq)])

    all_intermed = np.concatenate((intermed_preds, intermed_upreds))
    all_labels = pred_labels + un_seq.true_labels

    umap_inter_all = UMAP(n_neighbors=30).fit_transform(all_intermed)
    plot_embedded(umap_inter_all, all_labels, all_groups, colors=colors).savefig(
        str(embedding_path / f"umap_intermediate_all.png"), dpi=300)

    tsne_inter_all = manifold.TSNE(perplexity=perplexity).fit_transform(all_intermed)
    plot_embedded(tsne_inter_all, all_labels, all_groups, colors=colors).savefig(
        str(embedding_path / f"tsne_intermediate_all_p{perplexity}.png"), dpi=300)

    # create som tsne for known and unknown data
    all_cases = validate.cases + udataset.cases

    case_data = []
    for case in all_cases:
        somdata = np.concatenate([
            case.get_tube(tube, kind="som").get_data().data
            for tube in model.config["tubes"]
        ], axis=2).flatten()
        case_data.append(somdata)
    case_data = np.array(case_data)

    perplexity = 50
    umap_som_all = UMAP(n_neighbors=30).fit_transform(case_data)
    plot_embedded(umap_som_all, all_labels, all_groups, colors=colors).savefig(
        str(embedding_path / f"umap_som_all.png"), dpi=300)

    tsne_som_all = manifold.TSNE(perplexity=perplexity).fit_transform(case_data)
    plot_embedded(tsne_som_all, all_labels, all_groups, colors=colors).savefig(
        str(embedding_path / f"tsne_som_all_p{perplexity}.png"), dpi=300)


if __name__ == "__main__":
    argmagic(main)
