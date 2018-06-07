import os
import logging
import json
from argparse import ArgumentParser
import datetime

import pandas as pd
import numpy as np

from .case_collection import CaseCollection
from .clustering import (
    create_pipeline,
    create_pipeline_multistage,
    create_presom_each,
    create_pre,
    create_pregate,
    create_pregate_nodes,
)
from .collection_transforms import ApplySingle, Merge
from .utils import (
    get_file_path, put_file_path,
    create_stamp,
    configure_print_logging
)
from .plotting import plot_overview, plot_history, create_plot, plot_upsampling

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

upsampled_num = args.upsampled if args.upsampled > 0 else None

outdir = "{}_{}".format(args.output, create_stamp())

tubes = list(map(int, args.tubes.split(";"))) if args.tubes else cases.tubes

groups = list(map(lambda x: x.strip(), args.groups.split(";"))) \
    if args.groups else cases.groups

pipeline_dict = {
    "normal": create_pipeline,
    "multi": create_pipeline_multistage,
}

pre_dict = {
    "normal": create_pre,
    "presom": create_presom_each,
    "pregate": create_pregate,
    "pregatesom": create_pregate_nodes,
    "": None,
}

if args.refnormal:
    train_groups = [g for g in groups if g != "normal"]
else:
    train_groups = groups

# create data views for input
train_view = cases.create_view(
    labels=refcases, tubes=tubes, num=num, groups=train_groups,
    bucketname=args.bucketname, tmpdir=args.temp
)
transform_view = cases.create_view(
    num=upsampled_num, tubes=tubes, groups=groups,
    bucketname=args.bucketname, tmpdir=args.temp
)

plotdir = os.path.join(args.plotdir, create_stamp())
os.makedirs(plotdir, exist_ok=True)

for tube in tubes:
    pipe = pipeline_dict[args.pipeline]()
    prefit = pre_dict[args.prefit]()
    pretrans = pre_dict[args.pretrans]()
    reduction = Merge(pipe, eachfit=prefit, eachtrans=pretrans)

    reduction.fit(train_view.yield_data(tube))

    fit_history = reduction.history["fit"]
    tubestr = "_tube{}".format(tube)
    if args.plot:
        create_plot(
            os.path.join(plotdir, "fit_each"+tubestr),
            lambda fig: plot_history(fig, fit_history, "Fit each")
        )

    df_all = reduction.transform(transform_view.yield_data(tube))

    trans_history = reduction.history["trans"]
    if args.plot:
        create_plot(
            os.path.join(plotdir, "transform_each"+tubestr),
            lambda fig: plot_history(fig, trans_history, "Transform each")
        )
        create_plot(
            os.path.join(plotdir, "histo_groups"+tubestr),
            lambda fig: plot_upsampling(fig, df_all)
        )

    def writefun(dest):
        df_all.to_csv(dest, sep=";")

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "tube{}.csv".format(tube))
    put_file_path(outpath, writefun, args.temp)
