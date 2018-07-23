import logging

from clustering.transformation.distance_classifier import DistanceClassifier
from clustering.collection import CaseCollection

from sklearn.metrics import confusion_matrix

GROUPS = ["CLL", "normal"]


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
    groups=["CLL", "normal"], num=20, infiltration=10.0
)

basic_test = cases.create_view(groups=["CLL", "normal"], num=100)

classifier = DistanceClassifier()

classifier.fit(basic_train.get_tube(1))

predictions = classifier.predict(basic_test.get_tube(1))
confusion = confusion_matrix(
    predictions["group"], predictions["prediction"], labels=GROUPS
)
print(confusion)
