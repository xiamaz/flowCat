import os
import logging
import json
from argparse import ArgumentParser
import datetime

import pandas as pd
import numpy as np

from .case_collection import CaseCollection
from .clustering import create_pipeline
from .utils import get_file_path, put_file_path

def create_stamp():
    stamp = datetime.datetime.now()
    return stamp.strftime("%Y%m%d_%H%M")

logging.basicConfig(level=logging.WARNING)

parser = ArgumentParser(
    prog="Clustering",
    description="Clustering preprocessing of flow cytometry data."
)
parser.add_argument("-o", "--output", help="Output file directory")
parser.add_argument("--refcases", help="Json with list of reference cases.")
parser.add_argument("--tubes", help="Selected tubes.")
parser.add_argument("-i", "--input", help="Input case json file.")
parser.add_argument("-t", "--temp", help="Temp path for file caching.")
parser.add_argument("--bucketname", help="S3 Bucket with data.")
args = parser.parse_args()

infojson_path = get_file_path(args.input, args.temp)

cases = CaseCollection(infojson_path, args.bucketname, args.temp)

pipe = create_pipeline()

refcases = []
if args.refcases:
    with open(args.refcases) as reffile:
        refcases = [
            case for cases in json.load(reffile).values() for case in cases
        ]

outdir = "{}_{}".format(args.output, create_stamp())

tubes = map(int, args.tubes.split(";")) if args.tubes else cases.tubes

for tube in tubes:
    if refcases:
        data = cases.get_train_from_case_labels(refcases, tube=tube)
    else:
        data = cases.get_train_data(num=5, tube=tube)
    pipe.fit(data)

    results = []
    labels = []
    groups = []
    for label, group, testdata in cases.get_all_data(num=2000, tube=tube):
        print("Upsampling {}".format(label))
        results.append(pipe.transform(testdata))
        labels.append(label)
        groups.append(group)
    df_all = pd.DataFrame(np.matrix(results))
    df_all["label"] = labels
    df_all["group"] = groups

    def writefun(dest):
        df_all.to_csv(dest, sep=";")

    outpath = os.path.join(outdir, "tube{}.csv".format(tube))
    put_file_path(outpath, writefun, args.temp)
