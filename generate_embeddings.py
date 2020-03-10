import numpy as np
import keras
from sklearn import manifold
from argmagic import argmagic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")

from flowcat.plots import embeddings as fc_embeddings
from flowcat.classifier import SOMClassifier
from flowcat import utils, io_functions, mappings


def create_intermediate_model(model, dest_layer="concatenate_1"):
    """Create a new model from the given model with output on an intermediary layer."""
    intermediate_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(dest_layer).output
    )

    return intermediate_model


def generate_intermediate_embeddings(data, model):
    """Create intermediate embeddings for the given data."""
    seq = model.create_sequence(data)
    intermed = create_intermediate_model(model.model, dest_layer="concatenate_1")
    embeds = np.array([p for p in intermed.predict_generator(seq)])
    return embeds


def reshape_som_data(data):
    """Reshape SOM data for dimensionality reduction."""


def intermediate_tsne(dataset, model, output):
    # generate and transform intermediate embeddings
    embeddings = generate_intermediate_embeddings(dataset, model)
    tsne_embeddings = manifold.TSNE(perplexity=50).fit_transform(embeddings)

    labels = dataset.groups

    fig, ax = plt.subplots(figsize=(6, 6))
    fc_embeddings.plot_embedding(ax, tsne_embeddings, labels, mappings.ALL_GROUP_COLORS)
    print(embeddings)

    fig.tight_layout()
    fig.savefig(
        str(output / f"intermediate_tsne.png"),
        dpi=300,
    )


def som_tsne(dataset, model, output):
    """SOM data tsne output."""
    # concatenate SOM dataset into a single numpy array
    embeddings, _ = model.array_from_cases(dataset)
    for tube in range(3):
        t_embeddings = embeddings[tube]
        n, _, _, y = t_embeddings.shape
        t_embeddings = t_embeddings.reshape((n, -1))

        tsne_embeddings = manifold.TSNE(perplexity=50).fit_transform(t_embeddings)

        labels = dataset.groups
        fig, ax = plt.subplots(figsize=(6, 6))
        fc_embeddings.plot_embedding(ax, tsne_embeddings, labels, mappings.ALL_GROUP_COLORS)
        print(embeddings)

        fig.tight_layout()
        fig.savefig(
            str(output / f"som_tsne_{tube}.png"),
            dpi=300,
        )

    raise RuntimeError

    embeddings = np.concatenate(embeddings, axis=-1)
    n, _, _, y = embeddings.shape

    embeddings = embeddings.reshape((n, -1))

    tsne_embeddings = manifold.TSNE(perplexity=50).fit_transform(embeddings)

    labels = dataset.groups
    fig, ax = plt.subplots(figsize=(6, 6))
    fc_embeddings.plot_embedding(ax, tsne_embeddings, labels, mappings.ALL_GROUP_COLORS)
    print(embeddings)

    fig.tight_layout()
    fig.savefig(
        str(output / f"som_tsne.png"),
        dpi=300,
    )

def main(
        data: utils.URLPath = None,
        model: utils.URLPath = None,
        preds: utils.URLPath = None,
        output: utils.URLPath = None,
):
    data = utils.URLPath("/data/flowcat-data/paper-cytometry/som/unused")
    dataset = io_functions.load_case_collection(data, data + ".json.gz")
    # output = utils.URLPath("/data/flowcat-data/paper-cytometry/tsne")
    output = utils.URLPath("teststuff_unused_style")
    output.mkdir()

    # predictions = io_functions.load_json(utils.URLPath("/data/flowcat-data/paper-cytometry/tsne/prediction.json"))
    model = SOMClassifier.load(utils.URLPath("/data/flowcat-data/paper-cytometry/classifier"))

    som_tsne(dataset, model, output)
    # intermediate_tsne(dataset, model, output)


argmagic(main)
