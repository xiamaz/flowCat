#!/usr/bin/env python3
'''
Neural network classification using keras
'''
# from typing import Callable
import os
import sys
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


RE_TUBE_NAME = re.compile(r"^tube(\d)(_\d)?\.csv$")


def exit_script(msg=""):
    '''Exit script. Add cleanup functions as needed.
    '''
    if msg:
        print(msg)
    print("Quitting script")
    sys.exit(0)


def handle_input(inputfunc):
    '''Implement input handler to avoid some common issues with format and
    method abort.'''
    while True:
        try:
            ret = inputfunc()
            return ret
        except ValueError:
            print("Please enter a number.")
            continue
        except IndexError:
            print("Enter a valid number in list.")
        except KeyboardInterrupt:
            # print a newline after ctrl-c
            print("\n")
            exit_script()


def evaluate(
        files: FilesDict,
        output: str,
        name: str,
        note: str,
        method: dict
) -> None:
    '''Evaluate upsampling data.'''
    data = UpsamplingData.from_files(files)
    # data.select_groups(["CLL", "normal", "CLLPL", "HZL",
    #                     "Mantel", "Marginal", "MBL"])
    # data.limit_size_to_smallest()
    clas = Classifier(data, name=name, output_path=output)

    clas.dump_experiment_info(note)
    for method_name, method_info in method.items():
        if method_name == "holdout":
            if method_info.startswith("a"):
                abs_num = int(method_info.strip("a"))
                clas.holdout_validation(abs_num=abs_num)
            elif method_info.startswith("r"):
                ratio = float(method_info.strip("r"))
                clas.holdout_validation(ratio=ratio)
        elif method_name == "kfold":
            clas.k_fold_validation(int(method_info))


def get_files(path: str, noninteractive: bool = False) -> FilesDict:
    '''Get upsampling files information.'''
    # get files
    dir_files = os.listdir(path)
    if not dir_files:
        raise RuntimeError("Empty input directory")
    csv_files = [f for f in dir_files if f.endswith(".csv")]
    if not csv_files:
        if noninteractive:
            exit_script("Empty input directory")
        print("No csv files in current dir. Select subdirectory.")

        def inputfunc():
            '''Get valid subpath or exit.'''
            print("\n".join(
                ["{}: {}".format(i, f) for i, f in enumerate(dir_files)]
                ))
            selected = int(input("Select project: "))
            subpath = os.path.join(path, dir_files[selected])
            return subpath
        return get_files(handle_input(inputfunc))

    files = defaultdict(list)
    for filename in csv_files:
        match = RE_TUBE_NAME.match(filename)
        if match is None:
            continue
        tube = match.group(1)
        files[tube].append(os.path.join(path, filename))
    return files


def get_arguments() -> (FilesDict, str, str, str, dict):
    '''Get arguments or use defaults for script execution.'''
    parser = ArgumentParser()
    parser.add_argument(
        "name",
        help="Experiment name"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input files location.",
        default="output/preprocess"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output location.",
        default="output/classification"
    )
    parser.add_argument(
        "-n", "--noninteractive",
        help="Run script without confirmation.",
        action="store_true"
    )
    parser.add_argument(
        "-m", "--method",
        help="Analysis method. Holdout or kfold",
        default="holdout:r0.8,kfold:10"
    )

    args = parser.parse_args()

    name = args.name or DEFAULT_NAME
    path = args.input or os.path.join("output", name)
    if not args.noninteractive:
        note = handle_input(lambda: input("Enter short note: "))
    else:
        note = NOTE

    method = dict(
        tuple(m.split(":", 2))
        for m in args.method.split(",")
    )
    return (
        get_files(path, args.noninteractive),
        args.output,
        name,
        note,
        method
    )


def main():
    '''Classification of single tubes
    '''
    files, output, name, note, method = get_arguments()
    # files = [("output/1_large_cohort_reduction/tube1.csv",
    #           "output/1_large_cohort_reduction/tube2.csv")]
    evaluate(files, output, name, note, method)


if __name__ == '__main__':
    main()
