#!/usr/bin/env python3
import sys
import os
import fcsparser
import random
from functools import reduce

import pandas as pd

INPATH = "cases"


def get_case_data(path, n=5):
    inputs = {}
    with open(path) as inputfile:
        for f in inputfile:
            cohort, filename = os.path.split(f.strip())
            inputs.setdefault(cohort, []).append(filename)

    selected = [os.path.join(INPATH, f) for v in inputs.values() for f in random.sample(v, n) ]

    input_fcs = [fcsparser.parse(f, data_set=0, encoding="latin-1")
                 for f in selected]

    # get names present in all lmd files
    names = reduce(lambda x, y: set(x) & set(y), [d.columns for _, d in input_fcs])
    data = pd.concat([
        f[list(names)] for _, f in input_fcs
    ])
    return data, names, inputs

def get_case(path, names):
    _, data = fcsparser.parse(os.path.join(INPATH, path), data_set=0, encoding="latin-1")
    return data[list(names)]

def load_test(cases, names):
    sel = random.sample([cc for c in cases.values() for cc in c], 1)[0]
    return get_case(sel, names)
