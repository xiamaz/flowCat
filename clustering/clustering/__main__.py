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
parser.add_argument(
    "--num", help="Number of selected cases", default=5, type=int
)
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
    num = None
else:
    num = args.num

outdir = "{}_{}".format(args.output, create_stamp())

tubes = map(int, args.tubes.split(";")) if args.tubes else cases.tubes

for tube in tubes:
    data = cases.get_train_data(num=num, tube=tube, labels=refcases)

    # concatenate all fcs files for refsom generation
    data = pd.concat([d for dd in data.values() for d in dd])

    pipe.fit(data)

    results = []
    labels = []
    groups = []
    for label, group, testdata in cases.get_all_data(num=300, tube=tube):
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
