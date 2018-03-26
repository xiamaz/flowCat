#!/usr/bin/env python3
'''
Neural network classification using keras
'''
# from typing import Callable
import os
import re
from argparse import ArgumentParser

from collections import defaultdict

from lib.upsampling import UpsamplingData
from lib.classification import Classifier
from lib.types import FilesDict
# from lib import plotting

# datatypes
DEFAULT_NAME = "tri_classif_0"
NOTE = "Testing visualization of neural network classifcations"


def evaluate(files: FilesDict, output: str, name: str, note: str) -> None:
    '''Evaluate upsampling data.'''
    data = UpsamplingData.from_files(files)
    # data.select_groups(["CLL", "normal", "CLLPL", "HZL",
    #                     "Mantel", "Marginal", "MBL"])
    # data.limit_size_to_smallest()
    clas = Classifier(data, name=name, output_path=output)
    # clas.holdout_validation(ratio=0.8)

    clas.dump_experiment_info(note)
    clas.k_fold_validation(k_num=10)


RE_TUBE_NAME = re.compile(r"^tube(\d)(_\d)?\.csv$")


def get_files(path: str) -> FilesDict:
    '''Get upsampling files information.'''
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    files = defaultdict(list)
    for filename in csv_files:
        match = RE_TUBE_NAME.match(filename)
        if match is None:
            continue
        tube = match.group(1)
        files[tube].append(os.path.join(path, filename))
    return files


def get_arguments() -> (FilesDict, str, str, str):
    '''Get arguments or use defaults for script execution.'''
    parser = ArgumentParser()
    parser.add_argument("name", help="Experiment name")
    parser.add_argument("-i", "--input", help="Input files location.")
    parser.add_argument("-o", "--output", help="Output location.",
                        default="output/classification")
    parser.add_argument("-n", "--noninteractive",
                        help="Run script without confirmation.",
                        action="store_true")

    args = parser.parse_args()

    name = args.name or DEFAULT_NAME
    path = args.input or os.path.join("output", name)
    if not args.noninteractive:
        note = input("Enter short note: ")
    else:
        note = NOTE

    return get_files(path), args.output, name, note


def main():
    '''Classification of single tubes
    '''
    files, output, name, note = get_arguments()
    # files = [("output/1_large_cohort_reduction/tube1.csv",
    #           "output/1_large_cohort_reduction/tube2.csv")]
    evaluate(files, output, name, note)


if __name__ == '__main__':
    main()
