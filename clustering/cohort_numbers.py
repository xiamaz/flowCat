#!/usr/bin/env python3
# Small script to output usable numbers of cases after filtering for tubes and
# marker identification
#
# Will also return marker composition
import os
from argparse import ArgumentParser
from urllib.parse import urlparse

from clustering.collection import CaseCollection

DESC = """Overview numbers for the given case collection in the specified
bucket.

Numbers are inferred from the provided case_info json files.
"""


def conv_tubes(raw):
    """Convert comma-separated list of numbers to a proper python list of ints.
    """
    return [int(r) for r in raw.split(",")]


parser = ArgumentParser(description=DESC)
parser.add_argument(
    "path", help="Path to the directory containing files.",
    default="s3://mll-flowdata",
)
parser.add_argument(
    "--tubes", help="Specify used tubes.",
    default="1,2",
    type=conv_tubes
)

args = parser.parse_args()

infopath = args.path if "case_info.json" in args.path else \
    os.path.join(args.path, "case_info.json")

collection = CaseCollection(infopath, args.tubes)

# simulate our clustering environment to get realistic numbers
all_view = collection.create_view()

print("Total: {}\n".format(len(all_view)))

print("Cohorts")
print("\n".join(
    ["{}: {}".format(k, len(v)) for k, v in all_view.groups.items()]
))

print("\n")

print("Materials")
for t in args.tubes:
    tube_view = all_view.get_tube(t)
    print("Tube {}".format(t))
    print("\n".join([
        "{}: {}".format(k, len(v))
        for k, v in tube_view.materials.items()
    ]))
    print("Total: {}".format(len(tube_view)))
    print("\n")

    print("Channels: {}".format(len(tube_view.markers)))
    print("\n".join(tube_view.markers))
    print("\n")

dissimilar_tubes = []
for single in all_view:
    if len(set(
            [single.get_tube(p).material for p in args.tubes]
    )) != 1:
        dissimilar_tubes.append(single)

print("Dissimilar: {}".format(len(dissimilar_tubes)))
print("\n".join([
    "{}|{}: {}".format(
        d.group, d.id, ", ".join([
            str(d.get_tube(p).material) for p in args.tubes
        ])
    ) for d in dissimilar_tubes
]))
