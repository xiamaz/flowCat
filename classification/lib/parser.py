import sys
import os
import re
import typing
from collections import defaultdict
from argparse import ArgumentParser

from .upsampling import ViewModifiers

RE_TUBE_NAME = re.compile(r"/tube(\d+)\.csv$")


FilesDict = typing.Dict[str, typing.Dict[str, str]]


def get_files(path: str, folder_match: str) -> FilesDict:
    '''Get upsampling files information.'''
    dirs = os.listdir(path)
    sel_dirs = [
        d for d in dirs if re.search(folder_match, d)
    ]
    all_files = {}
    for csvdir in sel_dirs:
        csv_files = [
            os.path.join(path, csvdir, f)
            for f in os.listdir(os.path.join(path, csvdir))
            if f.endswith(".csv")
        ]
        files: dict = defaultdict(dict)
        for filename in csv_files:
            match = RE_TUBE_NAME.search(filename)
            if match is None:
                continue
            tube = int(match.group(1))
            files[tube] = filename
        all_files[csvdir] = files
    return all_files


class CmdArgs:

    def __init__(self):
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
            "-o", "--output",
            help="Output location.",
            default="output/classification"
        )
        parser.add_argument(
            "--note",
            help="Adding a string note text describing the experiment."
        )
        parser.add_argument(
            "--method",
            help="Analysis method. Holdout or kfold. <name>:<prefix><num>",
            default="holdout:r0.8,kfold:10"
        )
        parser.add_argument(
            "--group",
            help="Groups included in analysis. Eg g1:CLL,MBL;normal"
        )
        parser.add_argument(
            "--pattern",
            help="Pattern to match for used data."
        )
        parser.add_argument(
            "--tubes",
            help="Selected tubes for processing",
        )
        parser.add_argument(
            "--modifiers",
            help=("Add global modifier options applied to data filtering. "
                  "smallest - Limit size of all cohorts to smallest cohort. ")
        )
        parser.add_argument(
            "--infiltration",
            help="Require minimum infiltration in training data.",
            default=0.0
        )
        parser.add_argument(
            "--size",
            help=("Add size information. In a semicolon delimited list. "
                  "Format: <group?>:(min-)max;... "
                  "Examples: CLL:100;normal:20-100;10-0 -- CLL max size "
                  "100, normal at least 20 to be included, all other cohorts "
                  "at least 10 cases to be included."),
            default=""
        )

        self.args = parser.parse_args()

    @property
    def note(self):
        return self.args.note if self.args.note else ""

    @property
    def cmd(self):
        cmd = __name__ + " " + " ".join(sys.argv)
        return cmd

    @property
    def output(self):
        return self.args.output

    @property
    def name(self):
        return self.args.name

    @property
    def methods(self):
        method = dict(
            tuple(m.split(":", 2))
            for m in self.args.method.split(",")
        )
        return method

    @property
    def infiltration(self):
        return float(self.args.infiltration)

    @property
    def files(self):
        pattern = self.args.pattern
        inputdir = self.args.input
        files = get_files(inputdir, pattern)
        return files

    @property
    def pattern(self):
        return self.args.pattern

    @property
    def groups(self):
        groups = [
            {
                "name": group[0],
                "tags": (
                    group[1].split(",")
                    if len(group) > 1
                    else [group[0]]
                ),
            }
            for group in
            [m.split(":") for m in self.args.group.split(";")]
        ] if self.args.group else []
        return groups

    @property
    def modifiers(self):
        modifiers = self.args.modifiers.split(";") \
            if self.args.modifiers else []
        return ViewModifiers.from_modifiers(modifiers)

    @property
    def sizes(self):
        """Format: g:min_max;
        """
        tokens = [symbol.split(":") for symbol in self.args.size.split(";")]
        size = {}
        for token in tokens:
            key, value = token if len(token) > 1 else ("", token[0])

            if key in size:
                raise RuntimeError("Duplicate key in size definition.")

            nums = value.split("-")

            # check whether argument is of form:
            # a-b, a- (min_value), or -b or b (max_value)
            if len(nums) == 2:
                min_value, max_value = nums
            elif value.endswith("-"):
                min_value, max_value = nums[0], 0
            else:
                min_value, max_value = 0, nums[0]

            size[key] = {
                "min_size": int(min_value) if min_value else 0,
                "max_size": int(max_value) if max_value else 0,
            }

        if "" not in size:
            size[""] = {
                "min_size": 0,
                "max_size": 0,
            }

        return size

    @property
    def tubes(self):
        if self.args.tubes:
            tubes = [int(s) for s in self.args.tubes.split(";")]
        else:
            tubes = []
        return tubes
