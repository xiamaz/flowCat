#!/usr/bin/env python3
'''
Neural network classification using keras
'''
# from typing import Callable
import os
import sys

from collections import defaultdict

import numpy as np

from classification.upsampling import DataCollection
from classification.classification import Classifier, NeuralNet, Tree
from classification.plotting import plot_combined
from classification.stamper import create_stamp
from classification.parser import CmdArgs
# from classification import plotting


def preprocess_data(
        data: DataCollection, args: CmdArgs
) -> "DataView":
    '''Apply grouping and other manipulation of the input data.'''

    for i, basedata in enumerate(data):
        view = basedata.filter_data(
            groups=args.groups,
            sizes=args.sizes,
            modifiers=args.modifiers,
        )
        yield i, view


def evaluate(
        args: CmdArgs
) -> None:
    '''Evaluate upsampling data.'''

    name = "{}_{}".format(args.name, create_stamp())

    data = DataCollection(args.files, args.pattern, args.tubes)

    results = []

    output_path = os.path.join(args.output, name)

    modelfunc = NeuralNet

    for i, view in preprocess_data(data, args):

        subname = "{}_{}".format(view.name, i)

        if args.transform == "sqrt":
            view = view.apply(np.sqrt)

        clas = Classifier(view, name=subname, output_path=output_path)

        for method_name, method_info in args.methods.items():
            if method_name == "holdout":
                if method_info.startswith("a"):
                    abs_num = int(method_info.strip("a"))
                    clas.holdout_validation(
                        modelfunc,
                        abs_num=abs_num,
                        infiltration=args.infiltration
                    )
                elif method_info.startswith("r"):
                    ratio = float(method_info.strip("r"))
                    clas.holdout_validation(
                        modelfunc,
                        ratio=ratio,
                        infiltration=args.infiltration
                    )
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
