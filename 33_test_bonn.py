"""
Test bonn data using previously generated model.
"""
from argparse import ArgumentParser
import flowcat

from keras import models


def load_model(path):
    binarizer = flowcat.utils.load_joblib(path / "binarizer.joblib")
    model = models.load_model(str(path / "model.h5"))
    return binarizer, model


def main(args):
    dataset = flowcat.SOMDataset.from_path(args.input)

    dataseq = flowcat.SOMSequence(dataset, binarizer, tube=1)
    preds = binarizer.inverse_transform(model.predict_generator(dataseq))
    result = dataset.labels == preds
    print("Untrained accuracy:", sum(result) / len(result))

    train, test = dataset.split(ratio=0.8)
    print(train, test)
    print(train.group_counts)
    print(test.group_counts)
    binarizer, model = load_model(args.model)

    trainseq = flowcat.SOMSequence(train, binarizer, tube=1)
    testseq = flowcat.SOMSequence(test, binarizer, tube=1)

    for layer in model.layers[1:]:
        layer.trainable = False
    model.fit_generator(
        generator=trainseq,
        epochs=30,
        validation_data=testseq)


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--input", type=flowcat.utils.URLPath)
    PARSER.add_argument("model", type=flowcat.utils.URLPath)
    main(PARSER.parse_args())
