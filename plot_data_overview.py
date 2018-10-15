# pylint: skip-file
# flake8: noqa
from flowcat.data import case_dataset

cases = case_dataset.CaseCollection.from_dir("s3://mll-flowdata/CLL-9F")

print(cases.get_markers(1))
