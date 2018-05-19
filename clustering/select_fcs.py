#!/usr/bin/env python3
import sys
import json
import random

sel = int(sys.argv[1])

def get_random_cases(cases, k=10):
    case_list = [
        cc for c in cases.values()
        for cc in random.sample(c, k)
    ]
    return case_list

with open("case_info.json") as jsfile:
    cases = json.load(jsfile)

# select specific cohorts
cases = {k: cases[k] for k in ["CLL", "normal"]}

selection = get_random_cases(cases, 100)

same_tube_sel = [
    t["path"]
    for s in selection
    for t in s["destpaths"]
    if t["tube"] == sel
]

with open("input.lst", "w") as lstfile:
    lstfile.write("\n".join(same_tube_sel))
