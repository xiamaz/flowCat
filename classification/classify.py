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
from lib.classification import Classifier, NeuralNet, Tree
from lib.types import FilesDict
from lib.plotting import plot_combined
from lib.stamper import create_stamp
# from lib import plotting

# datatypes
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


def preprocess_data(
        files: FilesDict,
        groups: list,
        data_filters: list,
        tubes: list
) -> UpsamplingData:
    '''Apply grouping and other manipulation of the input data.'''

    data = UpsamplingData.from_files(files, tubes)

    cutoff = 0
    max_size = 0
    sizes = []
    iter_group = ""
    for option in data_filters:
        if option == "smallest":
            max_size = min(data.get_group_sizes())
        elif "max_size" in option:
            max_size = int(option.lstrip("max_size:"))
        elif "iter_size" in option:
            # used in conjunction with max_size
            iter_group, max_group, step = option.lstrip(
                "iter_size:").split(",")
            sizes = range(max_size, int(max_group)+1, int(step))

    if not sizes:
        sizes = [max_size]

    for size in sizes:
        if iter_group:
            size = {
                g: size if g == iter_group else max_size
                for g in data.get_group_names()
            }
        view = data.filter_data(groups=groups, cutoff=cutoff, max_size=size)
        yield size, view


def evaluate(
        file_data: dict,
        output: str,
        name: str,
        info_args: str,
        method: dict,
) -> None:
    '''Evaluate upsampling data.'''

    name = "{}_{}".format(name, create_stamp())

    results = []

    output_path = os.path.join(output, name)

    modelfunc = NeuralNet

    for i, view in preprocess_data(**file_data):
        subname = "{}_{}".format(name, i)
        clas = Classifier(view, name=subname, output_path=output_path)

        for method_name, method_info in method.items():
            if method_name == "holdout":
                if method_info.startswith("a"):
                    abs_num = int(method_info.strip("a"))
                    clas.holdout_validation(modelfunc, abs_num=abs_num)
                elif method_info.startswith("r"):
                    ratio = float(method_info.strip("r"))
                    clas.holdout_validation(modelfunc, ratio=ratio)
            elif method_name == "kfold":
                clas.k_fold_validation(modelfunc, int(method_info))

        clas.dump_experiment_info(**info_args)
        results.append((i, clas.past_experiments))

    plot_combined(results, output_path)


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

    files = defaultdict(dict)
    for filename in csv_files:
        match = RE_TUBE_NAME.match(filename)
        if match is None:
            continue
        tube = int(match.group(1))
        seq = int(match[2]) if match[2] else 0
        files[seq][tube] = os.path.join(path, filename)
    return files


def get_arguments() -> (FilesDict, str, str, str, dict):
    '''Get arguments or use defaults for script execution.'''
    parser = ArgumentParser()
    parser.add_argument(
        "name",
        help="Experiment name",
    )
    parser.add_argument(
        "-i", "--input",
        help="Input files location.",
        default="output/preprocess"
    )
    parser.add_argument(
        "-n", "--note",
        help="Adding a string note text describing the experiment."
    )
    parser.add_argument(
        "-o", "--output",
        help="Output location.",
        default="output/classification"
    )
    parser.add_argument(
        "-x", "--noninteractive",
        help="Run script without confirmation.",
        action="store_true"
    )
    parser.add_argument(
        "-m", "--method",
        help="Analysis method. Holdout or kfold. <name>:<prefix><num>",
        default="holdout:r0.8,kfold:10"
    )
    parser.add_argument(
        "-g", "--group",
        help="Groups included in analysis. Eg CLL;normal or g1:CLL,MBL;normal"
    )
    parser.add_argument(
        "--tubes",
        help="Selected tubes for processing",
    )
    parser.add_argument(
        "-f", "--filters",
        help=("Add data preprocessing options. "
              "smallest - Limit size of all cohorts to smallest cohort. "
              "max_size - Set highest size to maximum size.")
    )

    args = parser.parse_args()

    # additional info
    if not args.note:
        note = handle_input(lambda: input("Enter short note: "))
    else:
        note = args.note
    # get cmd line
    cmd = __name__ + " " + " ".join(sys.argv)
    additional = {"note": note, "cmd": cmd}

    name = args.name

    method = dict(
        tuple(m.split(":", 2))
        for m in args.method.split(",")
    )
    files = get_files(
        args.input or os.path.join("output", name),
        args.noninteractive
    )

    groups = [
        {
            "name": group[0],
            "tags": (
                group[1].split(",")
                if len(group) > 1
                else [group[0]]
            )
        }
        for group in
        [m.split(":") for m in args.group.split(";")]
    ] if args.group else []

    filters = args.filters.split(";") if args.filters else []

    if args.tubes:
        tubes = [int(s) for s in args.tubes.split(";")]
    else:
        tubes = []

    options = {
        "name": name,
        "method": method,
        "output": args.output,
        "file_data": {
            "files": files,
            "groups": groups,
            "data_filters": filters,
            "tubes": tubes,
        },
        "info_args": additional,
    }
    return options


def main():
    '''Classification of single tubes
    '''
    evaluate(**get_arguments())


if __name__ == '__main__':
    main()
