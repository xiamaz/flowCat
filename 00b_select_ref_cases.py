#!/usr/bin/env python3
# pylint: skip-file
# flake8: noqa
"""
Output a json list of case ids.
"""
import inspect
import typing
import math
from argparse import ArgumentParser

import flowcat
from flowcat import io_functions
from flowcat.dataset.case_dataset import CaseCollection


def function_arg_descs(fun) -> dict:
    """Return a dict of function argument to description from the docstring."""
    docstring = [l.strip() for l in fun.__doc__.split("\n")]
    args_start = docstring.index("Args:")
    try:
        return_index = docstring.index("Returns:")
    except ValueError:
        return_index = None
    try:
        raises_index = docstring.index("Raises:")
    except ValueError:
        raises_index = None

    if return_index and raises_index:
        args_end = min(return_index, raises_index)
    else:
        args_end = return_index or raises_index
    args = [l.split(":", 1) for l in docstring[args_start+1: args_end]]
    arg_map = {
        name.strip(): desc.strip() for name, desc in filter(lambda x: len(x) == 2, args)
    }
    return arg_map


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
        res = [None if token in ("None", "") else itemtype(token) for itemtype, token in zip(itemtypes, tokens)]
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
    filter_descs = function_arg_descs(flowcat.dataset.case.filter_case)
    for name, fun in argfuns.items():
        parser.add_argument(f"--{name}", type=fun, help=filter_descs.get(name, None))
    return parser


def print_cases(dataset: CaseCollection):
    """Print cases in a table."""
    print(f"{'Label':<40}\t{'Group':<6}\t{'Infiltration':<4}")
    for case in dataset:
        print(f"{case.id:<40}\t{case.group:<6}\t{case.infiltration:<4}")


PARSER = ArgumentParser(
    usage="Output a list of case ids based on the parameters")
PARSER.add_argument(
    "--data", type=flowcat.utils.URLPath)
PARSER.add_argument(
    "--meta", type=flowcat.utils.URLPath)
PARSER.add_argument(
    "--sample", type=int, help="Stratified sample of cases.")
filter_args = PARSER.add_argument_group("Filter arguments")
add_filter_args(filter_args)
PARSER.add_argument(
    "output",
    type=flowcat.utils.URLPath,
    help="Output destination for newly filtered results")


args = PARSER.parse_args()

cases = io_functions.load_case_collection(args.data, args.meta)
print(cases)

input_args = {n: getattr(args, n) for n in filter_signature()}
filtered, _ = cases.filter_reasons(**input_args)

filtered = filtered.sample(args.sample)

print("Found cases", filtered)
print_cases(filtered)

found_ids = filtered.labels
flowcat.io_functions.save_json(found_ids, args.output)
