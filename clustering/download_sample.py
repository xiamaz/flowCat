#!/usr/bin/env python3
import argparse
import pathlib

from clustering import collection as ccollection
from clustering import utils as cutils


DESC = "Download a random sample of the given dataset in the bucket."


def get_args():
    """Get commandline arguments for configuration of input and output path."""
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument(
        "--path", default="s3://mll-flowdata/CLL-9F",
        help="Path to the input reference data."
    )
    parser.add_argument(
        "--num", default="10", type=int,
        help="Number of cases per cohort."
    )
    parser.add_argument(
        "--groups", default="CLL,normal,MBL", type=lambda s: s.split(","),
        help="Included groups."
    )
    parser.add_argument(
        "output", default="selection", type=pathlib.Path,
        help="Output directory for selected cases."
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # override the tmp path
    cutils.TMP_PATH = args.output

    coll = ccollection.CaseCollection(args.path, tubes=[1, 2])
    selected_cases = coll.create_view(num=args.num, groups=args.groups)

    cutils.save_json(
        cutils.TMP_PATH / "sel_case_info.json", selected_cases.json
    )

    selected_cases.download_all()


if __name__ == "__main__":
    main()
