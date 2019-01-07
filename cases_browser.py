#!/usr/bin/env python3
"""
Commandline browser of a flowCat dataset.

Ability to output lists of case ids and other properties to ease
generating static configurations (eg not regenerated in each run),
which makes searching for bugs much easier.
"""
import argparse
import cmd

from flowcat import configuration, utils
from flowcat.dataset import case_dataset


def tokenize(arg):
    "Structure of query tokens: <g|l>::<query>"
    cmds = arg.split(" > ")
    if len(cmds) == 1:
        search, save = cmds[0], None
    elif len(cmds) == 2:
        search, save = cmds
    else:
        raise RuntimeError

    queries = search.split()
    tokens = []
    for query in queries:
        args = query.split("::")
        tokens.append({
            "type": args[0],
            "query": args[1],
        })
    return tokens, save


def info_case(case):
    casestr = f"{case.id} {case.group} {case.infiltration} {case.sureness}"
    print(casestr)


class Interpreter(cmd.Cmd):
    """Commandline interpreter."""
    intro = "Explore and extract dataset information."
    prompt = "> "

    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = utils.URLPath(path)
        self.data = case_dataset.CaseCollection.from_path(path)

    def do_info(self, arg):
        "Output basic information on dataset"
        print(self.data)

    def do_ls(self, arg):
        "List members of groups or cases."
        if arg == "":
            group_counts = self.data.group_count
            for name, count in group_counts.items():
                print(f"{name}: {count} cases")
        else:
            queries, save = tokenize(arg)

            cases = list(self.data.data)
            for query in queries:
                if query["type"] == "g":
                    cases = [c for c in cases if c.group == query["query"]]
                elif query["type"] == "ig":
                    cases = [
                        c for c in cases if c.infiltration > float(query["query"])]
                elif query["type"] == "il":
                    cases = [
                        c for c in cases if c.infiltration < float(query["query"])]
                elif query["type"] == "s":
                    cases = [
                        c for c in cases if c.infiltration == int(query["query"])]
                else:
                    print("Invalid type ", query["type"])
            [info_case(c) for c in cases[:10]]
            print(f"Total {len(cases)}")
            if save:
                print(f"Saving labels to {save}")
                labels = [c.id for c in cases]
                utils.save_json(labels, save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")

    args = parser.parse_args()

    interpreter = Interpreter(args.cases)
    interpreter.cmdloop()
