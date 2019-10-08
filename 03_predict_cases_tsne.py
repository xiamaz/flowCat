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
from tensorflow import keras
from argmagic import argmagic

from flowcat import io_functions, utils, som_dataset


def load_model(path: utils.URLPath):
    """Load a classification model and accompanying binarizer."""
    config = io_functions.load_json(path / "config.json")
    model = keras.models.load_model(str(path / "model.h5"))
    binarizer = io_functions.load_joblib(path / "binarizer.joblib")
    return model, binarizer, config


def main(data: utils.URLPath, model: utils.URLPath, output: utils.URLPath):
    data, model = map(utils.URLPath, (
        "output/0-munich-data/som-minmax",
        "output/simple-classifiers/test-conv-32-9class",
    ))
    output = utils.URLPath("output/model-analysis")
    labels = io_functions.load_json(model / "ids_validate.json")
    dataset = io_functions.load_case_collection(data, data + ".json")
    validate = dataset.filter(labels=labels)

    model, binarizer, config = load_model(model)

    def getter(data, tube):
        return data.get_tube(tube, kind="som").get_data().data

    val_seq = som_dataset.SOMSequence(
        validate, binarizer,
        get_array_fun=getter,
        tube=config["tubes"],
        batch_size=128,
        pad_width=config["pad_width"])

    preds = np.array([p for p in model.predict_generator(val_seq)])
    trues = np.concatenate([s for _, s in val_seq])

    roc_path = output / "roc"
    roc_path.mkdir()

    curves = {}
    for i, group in enumerate(config["groups"]):
        curves[group] = metrics.roc_curve(trues[:, i], preds[:, i])

    auc = {}
    for i, group in enumerate(config["groups"]):
        auc[group] = metrics.roc_auc_score(trues[:, i], preds[:, i])

    macro_auc = metrics.roc_auc_score(trues, preds, average="macro")
    micro_auc = metrics.roc_auc_score(trues, preds, average="micro")
    io_functions.save_json(
        {
            "one-vs-rest": auc,
            "macro": macro_auc,
            "micro": micro_auc,
        },
        roc_path / "auc.json")

    fig, ax = plt.subplots()
    for name, curve in curves.items():
        ax.plot(curve[0], curve[1], label=name)

    ax.plot((0, 1), (0, 1), "k--")
    ax.legend()
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC one-vs-rest")

    fig.tight_layout()
    fig.savefig(str(roc_path / "roc.png"), dpi=300)
    plt.close()

    # calculate accuracy for a certain certainty
    # how about w score above 0.95?
    thres_path = output / "threshold"
    thres_path.mkdir()
    threshold_results = []
    for threshold in np.arange(0.25, 1.0, 0.05):
        index_above = np.argwhere(np.any(preds > threshold, axis=1)).squeeze()
        sel_preds = preds[index_above, :]
        sel_trues = trues[index_above, :]
        pred_labels = binarizer.inverse_transform(sel_preds)
        true_labels = binarizer.inverse_transform(sel_trues)
        findex = np.argwhere(pred_labels != true_labels).squeeze()
        false_indexes = index_above[findex]
        included = len(index_above) / len(preds)
        acc = metrics.accuracy_score(true_labels, pred_labels)
        print(threshold, included, acc)
        threshold_results.append((threshold, included, acc))
    io_functions.save_json(threshold_results, thres_path / "thresholds.json")

    tarr = np.array(threshold_results)
    fig, ax = plt.subplots()
    ax.plot(tarr[:, 0], tarr[:, 1], label="included")
    ax.plot(tarr[:, 0], tarr[:, 2], label="acc")
    ax.legend()
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Classification accuracy / Included cases ratio")
    fig.savefig(str(thres_path / "threshold.png"), dpi=300)

    origmeta = io_functions.load_json(
        utils.URLPath(
            "/data/flowcat-data/mll-flowdata/CLL-9F/case_info_2018-09-13.json"
        ))

    false_labels = np.array(validate.labels)[false_indexes]
    for index, label in zip(findex, false_labels):
        case = validate.get_label(label)
        print(case, pred_labels[index])
        print(case.diagnosis)
        for m in origmeta:
            if m["id"] == label:
                print(m["sureness"])

    # tsne of result vectors
    embedding_path = output / "embedding-preds"
    embedding_path.mkdir()

    from sklearn import manifold
    perplexity = 50
    tsne_pred_model = manifold.TSNE(perplexity=perplexity)
    tsne_preds = tsne_pred_model.fit_transform(preds)

    fig, ax = plt.subplots(figsize=(12, 8))
    all_pred_labels = val_seq.true_labels
    for group in config["groups"]:
        sel_dots = tsne_preds[np.array([i == group for i in all_pred_labels]), :]
        ax.scatter(sel_dots[:, 0], sel_dots[:, 1], label=group, s=16, marker="o")
    ax.legend()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"TSNE (perplexity={perplexity})")
    fig.savefig(str(embedding_path / "tsne_validation_p50.png"), dpi=300)
    plt.close()

    # tsne of intermediate layers
    intermediate_model = keras.Model(
        inputs=model.input, outputs=model.get_layer("concatenate").output)

    intermed_preds = np.array([p for p in intermediate_model.predict_generator(val_seq)])

    perplexity = 30
    tsne_inter_preds = manifold.TSNE(perplexity=perplexity).fit_transform(intermed_preds)

    fig, ax = plt.subplots(figsize=(12, 8))
    all_pred_labels = val_seq.true_labels
    for group in config["groups"]:
        sel_dots = tsne_inter_preds[np.array([i == group for i in all_pred_labels]), :]
        ax.scatter(sel_dots[:, 0], sel_dots[:, 1], label=group, s=16, marker="o")
    ax.legend()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"TSNE (perplexity={perplexity})")
    fig.savefig(str(embedding_path / "tsne_intermediate_validation_p30.png"), dpi=300)
    plt.close()


    # unknown data
    udata = utils.URLPath("output/unknown-cohorts-processing/som/som")
    udataset = io_functions.load_case_collection(udata, udata + ".json")
    udataset.set_data_path(utils.URLPath(""))
    un_seq = som_dataset.SOMSequence(
        udataset, binarizer,
        get_array_fun=getter,
        tube=config["tubes"],
        batch_size=128,
        pad_width=config["pad_width"])
    upreds = np.array([p for p in model.predict_generator(un_seq)])
    intermed_upreds = np.array([p for p in intermediate_model.predict_generator(un_seq)])

    binarizer.inverse_transform(upreds)

    tsne_upreds = manifold.TSNE(perplexity=30).fit_transform(upreds)
    fig, ax = plt.subplots(figsize=(12, 8))
    all_pred_labels = un_seq.true_labels
    for group in set(udataset.groups):
        sel_dots = tsne_upreds[np.array([i == group for i in all_pred_labels]), :]
        ax.scatter(sel_dots[:, 0], sel_dots[:, 1], label=group, s=16, marker="o")
    ax.legend()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"TSNE (perplexity={perplexity})")
    fig.savefig(str(embedding_path / "tsne_unknown_p30.png"), dpi=300)
    plt.close()

    allpreds = np.concatenate((preds, upreds))
    tsne_allpreds = manifold.TSNE(perplexity=30).fit_transform(allpreds)
    fig, ax = plt.subplots(figsize=(12, 8))
    all_pred_labels = val_seq.true_labels + un_seq.true_labels
    groups = set(udataset.groups + dataset.groups)
    colors = sns.cubehelix_palette(len(groups), rot=4, dark=0.30)
    for i, group in enumerate(groups):
        sel_dots = tsne_allpreds[np.array([i == group for i in all_pred_labels]), :]
        ax.scatter(sel_dots[:, 0], sel_dots[:, 1], label=group, s=16, marker="o", c=[colors[i]])
    ax.legend()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"TSNE (perplexity={perplexity})")
    fig.savefig(str(embedding_path / "tsne_all_p30.png"), dpi=300)
    plt.close()

    intermed_allpreds = np.concatenate((intermed_preds, intermed_upreds))
    tsne_intermed_allpreds = manifold.TSNE(perplexity=30).fit_transform(intermed_allpreds)
    fig, ax = plt.subplots(figsize=(12, 8))
    all_pred_labels = val_seq.true_labels + un_seq.true_labels
    groups = set(udataset.groups + dataset.groups)
    colors = sns.cubehelix_palette(len(groups), rot=4, dark=0.30)
    for i, group in enumerate(groups):
        sel_dots = tsne_intermed_allpreds[np.array([i == group for i in all_pred_labels]), :]
        ax.scatter(sel_dots[:, 0], sel_dots[:, 1], label=group, s=16, marker="o", c=[colors[i]])
    ax.legend()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"TSNE (perplexity={perplexity})")
    fig.savefig(str(embedding_path / "tsne_all_intermed_p30.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    argmagic(main)
