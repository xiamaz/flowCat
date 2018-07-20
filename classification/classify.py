#!/usr/bin/env python3
'''
Neural network classification using keras
'''
# from typing import Callable
import os
import sys

from collections import defaultdict

from lib.upsampling import UpsamplingData
from lib.classification import Classifier, NeuralNet, Tree
from lib.plotting import plot_combined
from lib.stamper import create_stamp
from lib.parser import CmdArgs
# from lib import plotting


def preprocess_data(
        args: CmdArgs
) -> UpsamplingData:
    '''Apply grouping and other manipulation of the input data.'''

    data = UpsamplingData.from_files(args.files, args.tubes)

    cutoff = 0
    max_size = 0
    sizes = []
    iter_group = ""
    for option in args.filters:
        if option == "smallest":
            max_size = min(data.get_group_sizes())
        elif "max_size" in option:
            max_size = int(option.lstrip("max_size:"))
        elif "iter_size" in option:
            # used in conjunction with max_size
            iter_group, max_group, step = option.lstrip(
                "iter_size:").split(",")
            sizes = range(max_size, int(max_group) + 1, int(step))

    if not sizes:
        sizes = [max_size]

    for size in sizes:
        for name, view in data.filter_data(
                groups=args.groups, cutoff=cutoff, max_size=size
        ):
            if iter_group:
                size = {
                    g: size if g == iter_group else max_size
                    for g in view.group_names()
                }
            vname = "{}_{}".format(name, size)
            yield vname, view


def evaluate(
        args: CmdArgs
) -> None:
    '''Evaluate upsampling data.'''

    name = "{}_{}".format(args.name, create_stamp())

    results = []

    output_path = os.path.join(args.output, name)

    modelfunc = NeuralNet

    for i, view in preprocess_data(args):
        subname = "{}_{}".format(name, i)
        clas = Classifier(view, name=subname, output_path=output_path)

        for method_name, method_info in args.methods.items():
            if method_name == "holdout":
                if method_info.startswith("a"):
                    abs_num = int(method_info.strip("a"))
                    clas.holdout_validation(modelfunc, abs_num=abs_num)
                elif method_info.startswith("r"):
                    ratio = float(method_info.strip("r"))
                    clas.holdout_validation(modelfunc, ratio=ratio)
            elif method_name == "kfold":
                clas.k_fold_validation(modelfunc, int(method_info))

        clas.dump_experiment_info(note=args.note, cmd=args.cmd)
        results.append((i, clas.past_experiments))

    plot_combined(results, output_path)


def main():
    '''Classification of single tubes
    '''
    evaluate(CmdArgs())


if __name__ == '__main__':
    main()
