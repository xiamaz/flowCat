import sys

import logging

import pandas as pd

from sklearn.metrics import confusion_matrix

from clustering.transformation.distance_classifier import DistanceClassifier
from clustering.collection import CaseCollection

sys.path.append("../classification")
from classification import plotting as cl_plotting


GROUPS = [
    # "CLL", "CLLPL", "FL", "HZL", "LPL", "MBL", "Mantel", "Marginal", "normal"
    "CLL", "MBL", "normal"
]


def configure_print_logging(rootname="clustering"):
    """Configure default logging for visual output to stdout."""
    rootlogger = logging.getLogger(rootname)
    rootlogger.setLevel(logging.INFO)
    formatter = logging.Formatter()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)


def main():
    configure_print_logging()

    cases = CaseCollection(inputpath="s3://mll-flowdata/CLL-9F", tubes=[1, 2])

    basic_train = cases.create_view(groups=GROUPS, num=20, infiltration=10.0)

    basic_test = cases.create_view(groups=GROUPS, num=100)

    classifier = DistanceClassifier()

    classifier.fit(basic_train.get_tube(1))

    tubeview = basic_test.get_tube(1)
    predictions = classifier.predict(tubeview)
    confusion = confusion_matrix(
        predictions["group"], predictions["prediction"], labels=GROUPS
    )
    cl_plotting.plot_confusion_matrix(
        confusion, classes=GROUPS,
        normalize=True,
        filename="distance_conf", dendroname="distance_dendro"
    )

    records = [
        {**t.result, **t.metainfo_dict} for t in tubeview.data
        if t.result_success
    ]
    alldata = pd.DataFrame.from_records(records)
    alldata.to_csv("dist_predictions.csv")

    confusiondf = pd.DataFrame(data=confusion, columns=GROUPS, index=GROUPS)
    confusiondf.to_csv("dist_confusion.csv")


if __name__ == "__main__":
    main()
