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


LOGGER = logging.getLogger(__name__)


class CaseCollection:

    def __init__(self, infopath, bucketname, tmpdir="tmp"):
        with open(infopath) as ifile:
            self._data = json.load(ifile)

        self.tubes = self._unique_tubes()
        self.markers = self._majority_markers()
        self.tmpdir = tmpdir
        self.s3 = boto3.resource("s3")
        self.bucket = self.s3.Bucket(bucketname)

    def _unique_tubes(self):
        tubes = [
            set([d["tube"] for d in c["destpaths"]])
            for cas in self._data.values() for c in cas
        ]
        return reduce(lambda x, y: x | y, tubes)

    def _majority_markers(self, threshold=0.9):
        tube_markers = {}
        for cases in self._data.values():
            for case in cases:
                for filepath in case["destpaths"]:
                    tube_markers.setdefault(filepath["tube"], []).extend(
                        filepath["markers"]
                    )
        marker_counts = {
            t: collections.Counter(m) for t, m in tube_markers.items()
        }
        marker_ratios = {
            t: {k: c[k]/len(self._data) for k in c}
            for t, c in marker_counts.items()
        }
        return {
            t: [v for v, r in c.items() if r >= threshold]
            for t, c in marker_ratios.items()
        }

    @staticmethod
    def _limit_groups(data, groups):
        filtered_data = {g: data[g] for g in groups}
        return filtered_data

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
            os.makedirs(os.path.split(destpath)[0], exist_ok=True)
            self.bucket.download_file(key, destpath)

        _, data = fcsparser.parse(destpath, data_set=0, encoding="latin-1")
        try:
            selected = data[self.markers[tube]]
        except KeyError:
            selected = None
        return selected

    def get_train_data(self, labels=None, num=5, groups=None, tube=1):
        # limit to groups
        data = self._data

        if groups:
            data = self._limit_groups(data, groups)

        # randomly sample num cases from each group
        if labels:
            data = {
                cohort: [case for case in cases if case["id"] in labels]
                for cohort, cases in data.items()
            }

        if num:
            data = {
                cohort: list(random.sample(cases, min(num, len(cases))))
                for cohort, cases in data.items()
            }

        train_fcs = {
            cohort: [
                f for f in
                [self._load_tube(s, tube) for s in cc] if f is not None
            ]
            for cohort, cc in data.items()
        }
        return train_fcs

    def get_all_data(self, num=None, groups=None, tube=1):
        # limit to groups
        data = self._data
        if groups:
            data = self._limit_groups(data, groups)

        for cohort, cases in data.items():
            if num:
                cases = random.sample(cases, min(num, len(cases)))
            for case in cases:
                fcsdata = self._load_tube(case, tube)
                if fcsdata is not None:
                    yield case["id"], cohort, fcsdata
