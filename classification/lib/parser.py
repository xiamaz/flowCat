import sys
import os
import re
import typing
from collections import defaultdict
from argparse import ArgumentParser

RE_TUBE_NAME = re.compile(r"tube(\d+)\.csv$")


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
            for f in os.listdir(os.path.join(path, csvdir)) if f.endswith(".csv")
        ]
        files = defaultdict(dict)
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
            "-n", "--note",
            help="Adding a string note text describing the experiment."
        )
        parser.add_argument(
            "-o", "--output",
            help="Output location.",
            default="output/classification"
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
            "--pattern",
            help="Pattern to match for used data."
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
    def files(self):
        pattern = self.args.pattern
        inputdir = self.args.input
        files = get_files(inputdir, pattern)
        return files

    @property
    def groups(self):
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
            [m.split(":") for m in self.args.group.split(";")]
        ] if self.args.group else []
        return groups

    @property
    def tubes(self):
        if self.args.tubes:
            tubes = [int(s) for s in self.args.tubes.split(";")]
        else:
            tubes = []
        return tubes

    @property
    def filters(self):
        filters = self.args.filters.split(";") if self.args.filters else []
        return filters
