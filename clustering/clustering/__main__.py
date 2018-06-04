import os
import logging
import json
from argparse import ArgumentParser
import datetime

import pandas as pd
import numpy as np

from .case_collection import CaseCollection
from .clustering import (
    create_pipeline, create_pipeline_multistage, create_presom_each, create_pre
)
from .collection_transforms import ApplySingle, Merge
from .utils import (
    get_file_path, put_file_path,
    create_stamp,
    configure_print_logging
)
from .plotting import plot_overview

from .cmd_args import create_parser

configure_print_logging()

parser = create_parser()
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

pipeline_dict = {
    "normal": create_pipeline,
    "multi": create_pipeline_multistage,
}

pre_dict = {
    "normal": create_pre,
    "presom": create_presom_each,
}

pipe = pipeline_dict[args.pipeline]()
pre = pre_dict[args.pre]()
reduction = Merge(pipe, eachfit=pre)

# create data views for input
train_view = cases.create_view(
    labels=refcases, tubes=tubes, num=num,
    bucketname=args.bucketname, tmpdir=args.temp
)
transform_view = cases.create_view(
    num=args.upsampled, tubes=tubes,
    bucketname=args.bucketname, tmpdir=args.temp
)

for tube in tubes:
    reduction.fit(train_view.yield_data(tube))

    df_all = reduction.transform(transform_view.yield_data(tube))

    def writefun(dest):
        df_all.to_csv(dest, sep=";")

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "tube{}.csv".format(tube))
    put_file_path(outpath, writefun, args.temp)
