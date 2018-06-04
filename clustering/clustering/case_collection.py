import sys
import os
import json
import logging
import random
import collections
from functools import reduce

import boto3
import pandas as pd

import fcsparser

from .case_transforms import MarkerFilter, TubesFilter


LOGGER = logging.getLogger(__name__)


class CaseView:
    def __init__(self, data, markers, bucketname="", tmpdir="tmp"):
        self._data = data
        self.s3 = None
        self.bucket = None
        if bucketname:
            self.s3 = boto3.resource("s3")
            self.bucket = self.s3.Bucket(bucketname)
        self.tmpdir = tmpdir
        self.markers = markers

    def _load_tube(self, case, tube):
        key = [
            path["path"] for path in case["destpaths"] if path["tube"] == tube
        ]
        if len(key) != 1:
            LOGGER.warning(
                "%s has %d entries for tube %d", case["id"], len(key), tube
            )
            return None

        key = key[-1]
        destpath = os.path.join(self.tmpdir, key)

        if not os.path.exists(destpath):
            if self.bucket:
                os.makedirs(os.path.split(destpath)[0], exist_ok=True)
                self.bucket.download_file(key, destpath)
            else:
                raise RuntimeError("File %s does not exist", destpath)

        _, data = fcsparser.parse(destpath, data_set=0, encoding="latin-1")
        try:
            selected = data[self.markers.selected_markers[tube]]
        except KeyError:
            selected = None
        return selected

    def yield_data(self, tube=1):
        for cohort, cases in self._data.items():
            for i, case in enumerate(cases):
                fcsdata = self._load_tube(case, tube)
                if fcsdata is not None:
                    LOGGER.info("Getting %s %d", cohort, i)
                    yield (
                        ("label", case["id"]),
                        ("group", cohort),
                        ("infiltration", case["infiltration"]),
                    ), fcsdata


class CaseCollection:

    def __init__(self, infopath):
        with open(infopath) as ifile:
            self._data = json.load(ifile)

        self.tubes = list(self._unique_tubes())
        # majority markers per tube
        self.markers = MarkerFilter(threshold=0.9)
        self.markers.fit(self._data)
        self._data = self.markers.transform(self._data)

    def _unique_tubes(self):
        tubes = [
            set([d["tube"] for d in c["destpaths"]])
            for cas in self._data.values() for c in cas
        ]
        return reduce(lambda x, y: x | y, tubes)

    @staticmethod
    def _limit_groups(data, groups):
        filtered_data = {g: data[g] for g in groups}
        return filtered_data

    def create_view(
            self, labels=None, num=None, groups=None, tubes=None, **kwargs
    ):
        """Filter view to specified criteria and return a new view object."""
        data = self._data
        if groups:
            data = self._limit_groups(data, groups)

        if tubes:
            filterer = TubesFilter(tubes=tubes, duplicate_allowed=False)
            data = filterer.transform(data)

        # randomly sample num cases from each group
        if labels:
            data = {
                cohort: [case for case in cases if case["id"] in labels]
                for cohort, cases in data.items()
            }

        if num:
            data = {
                cohort: random.sample(cases, min(num, len(cases)))
                for cohort, cases in data.items()
            }


        return CaseView(data, self.markers, **kwargs)
