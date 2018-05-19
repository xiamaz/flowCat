#!/usr/bin/env python3
import sys
import os
import fcsparser
import random
from functools import reduce

import pandas as pd

def get_case_data(path, n=5):
    with open(path) as inputfile:
        inputs = {
            k: v for k, v
            [os.path.split(f.strip()) for f in inputfile]
        }

    selected = [random.sample(n) for v in inputs.values()]

    input_fcs = [fcsparser.parse(f, data_set=0, encoding="latin-1")
                 for f in input_files]

    # get names present in all lmd files
    names = reduce(lambda x, y: set(x) & set(y), [d.columns for _, d in input_fcs])
    data = pd.concat([
        f[list(names)] for _, f in input_fcs
    ])
    return data, names, cases

def get_case(path, names):
    _, data = fcsparser.parse(f, data_set=0, encoding="latin-1")
    return data[names]
