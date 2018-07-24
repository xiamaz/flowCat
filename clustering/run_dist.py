import logging

import pandas as pd

from clustering.transformation.distance_classifier import DistanceClassifier
from clustering.collection import CaseCollection

from sklearn.metrics import confusion_matrix

GROUPS = [
    "CLL", "CLLPL", "FL", "HZL", "LPL", "MBL", "Mantel", "Marginal", "normal"
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


configure_print_logging()

cases = CaseCollection(
    infopath="s3://mll-flowdata/case_info.json", tubes=[1, 2]
)

basic_train = cases.create_view(
    groups=GROUPS, num=20, infiltration=10.0
)

basic_test = cases.create_view(groups=GROUPS, num=100)

classifier = DistanceClassifier()

classifier.fit(basic_train.get_tube(1))

tubeview = basic_test.get_tube(1)
predictions = classifier.predict(tubeview)
confusion = confusion_matrix(
    predictions["group"], predictions["prediction"], labels=GROUPS
)
print(confusion)

records = [
    {**t.result, **t.metainfo_dict} for t in tubeview.data
    if t.result_success
]
alldata = pd.DataFrame.from_records(records)
alldata.to_csv("dist_predictions.csv")

confusiondf = pd.DataFrame(data=confusion, columns=GROUPS, index=GROUPS)
confusiondf.to_csv("dist_confusion.csv")
