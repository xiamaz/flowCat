import os
import logging
import json
from argparse import ArgumentParser
import datetime

import pandas as pd
import numpy as np

from .case_collection import CaseCollection
from .clustering import (
    create_pipeline, create_pipeline_multistage, MarkerChannelFilter
)
from .utils import get_file_path, put_file_path

def create_stamp():
    stamp = datetime.datetime.now()
    return stamp.strftime("%Y%m%d_%H%M")

rootlogger = logging.getLogger("clustering")
rootlogger.setLevel(logging.INFO)
formatter = logging.Formatter()
handler = logging.StreamHandler()

handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

rootlogger.addHandler(handler)


parser = ArgumentParser(
    prog="Clustering",
    description="Clustering preprocessing of flow cytometry data."
)
parser.add_argument("-o", "--output", help="Output file directory")
parser.add_argument("--refcases", help="Json with list of reference cases.")
parser.add_argument("--tubes", help="Selected tubes.")
parser.add_argument(
    "--groups", help="Semicolon separated list of groups"
)
parser.add_argument(
    "--num", help="Number of selected cases", default=5, type=int
)
parser.add_argument(
    "--upsampled", help="Number of cases per cohort to be upsampled.",
    default=300, type=int
)
parser.add_argument("-i", "--input", help="Input case json file.")
parser.add_argument("-t", "--temp", help="Temp path for file caching.")
parser.add_argument("--bucketname", help="S3 Bucket with data.")
args = parser.parse_args()

infojson_path = get_file_path(args.input, args.temp)

cases = CaseCollection(infojson_path, )

refcases = []
if args.refcases:
    with open(args.refcases) as reffile:
        refcases = [
            case for cases in json.load(reffile).values() for case in cases
        ]
    num = None
else:
    num = args.num if args.num != -1 else None

outdir = "{}_{}".format(args.output, create_stamp())

tubes = list(map(int, args.tubes.split(";"))) if args.tubes else cases.tubes

pipe = create_pipeline()

train_view = cases.create_view(
    labels=refcases, tubes=tubes, num=num,
    bucketname=args.bucketname, tmpdir=args.temp
)
transform_view = cases.create_view(
    num=args.upsampled, tubes=tubes,
    bucketname=args.bucketname, tmpdir=args.temp
)

for tube in tubes:
    data = [data for _, _, data in train_view.yield_data(tube)]

    data = pd.concat(data)

    pipe.fit(data)

    results = []
    labels = []
    groups = []
    for label, group, testdata in transform_view.yield_data(tube=tube):
        print("Upsampling {}".format(label))
        upsampled = pipe.transform(testdata)
        results.append(upsampled)
        labels.append(label)
        groups.append(group)

    df_all = pd.DataFrame(np.matrix(results))
    df_all["label"] = labels
    df_all["group"] = groups

    def writefun(dest):
        df_all.to_csv(dest, sep=";")

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "tube{}.csv".format(tube))
    put_file_path(outpath, writefun, args.temp)
