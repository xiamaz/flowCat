#!/usr/bin/env python3
# pylint: skip-file
# flake8: noqa
"""
Output a json list of case ids.
"""
import inspect
import typing
from argparse import ArgumentParser

import flowcat


def filter_signature() -> dict:
    """Return the function signature for the filter cases function."""
    filter_sig = inspect.signature(flowcat.dataset.case.filter_case)
    filter_params = filter_sig.parameters
    filter_hints = typing.get_type_hints(flowcat.dataset.case.filter_case)
    filter_params = {
        name: filter_hints[name].__args__[0]
        for name, param in filter_params.items()
        if param.default is None
    }
    return filter_params


def to_list(itemtype):
    def parser(string):
        tokens = [itemtype(s.strip()) for s in string.split(",")]
        return tokens
    return parser


def to_tuple(itemtypes):
    def parser(string):
        tokens = [s.strip() for s in string.split(",")]
        res = [itemtype(token) for itemtype, token in zip(itemtypes, tokens)]
        return res
    return parser


def to_dict(itemtypes):
    keytype, valuetype = itemtypes
    def parser(string):
        tokens = [[ss.strip() for ss in s.strip().split(":")] for s in string.split(",")]
        res = {keytype(v[0]): valuetype(v[1]) for v in tokens}
        return res
    return parser


def map_signature_to_funs(sigs) -> dict:
    funs = {}
    for name, type_hint in sigs.items():
        if not isinstance(type_hint, typing._GenericAlias):
            funs[name] = type_hint
        elif type_hint.__origin__ == list:
            funs[name] = to_list(type_hint.__args__[0])
        elif type_hint.__origin__ == tuple:
            funs[name] = to_tuple(type_hint.__args__)
        elif type_hint.__origin__ == dict:
            funs[name] = to_dict(type_hint.__args__)
        else:
            print(f"Unsupported {type_hint}")
    return funs


def add_filter_args(parser):
    signature = filter_signature()
    argfuns = map_signature_to_funs(signature)
    for name, fun in argfuns.items():
        parser.add_argument(f"--{name}", type=fun)
    return parser


PARSER = ArgumentParser(
    usage="Output a list of case ids based on the parameters")
PARSER.add_argument(
    "-i", "--input",
    type=flowcat.utils.URLPath,
    required=True,
    help="Input fcs directory")
PARSER.add_argument(
    "-m", "--meta",
    type=flowcat.utils.URLPath,
    required=True,
    help="Input data meta without suffix")
filter_args = PARSER.add_argument_group("Filter arguments")
add_filter_args(filter_args)
PARSER.add_argument(
    "output",
    type=flowcat.utils.URLPath,
    help="Output destination for newly filtered results")


args = PARSER.parse_args()

cases = flowcat.CaseCollection.load(inputpath=args.input, metapath=args.meta)

input_args = {n: getattr(args, n) for n in filter_signature()}
filtered, _ = cases.filter_reasons(**input_args)

print("Found cases", filtered)

found_ids = filtered.labels
flowcat.utils.save_json(found_ids, args.output)
