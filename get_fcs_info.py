"""
Acquire FCS information needed for Miflowcyt document.

Also roughly check whether we have strongly diverging data in our dataset.
"""
from flowcat import dataset as fc_dataset, io_functions, utils
import fcsparser


def section(text, level=4, deco="#"):
    deco_text = deco * level
    section_text = f"{deco_text} {text} {deco_text}"
    print(section_text)


train_dataset = io_functions.load_case_collection(utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F"), utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/train.json.gz"))
test_dataset = io_functions.load_case_collection(utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F"), utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/test.json.gz"))

print("Loading all data used in paper analysis.")
dataset = train_dataset + test_dataset
print(dataset)

section("Get info for case 0")
case = dataset[0]
print(case)

sample = case.samples[0]
meta, data = fcsparser.parse(sample.complete_path)
for i in range(1, 13):
    name = f"$P{i}S"
    voltage = f"$P{i}V"
    print(meta[name], ":", meta[voltage])

import numpy as np
import pandas as pd
compensation_matrix = np.zeros((10, 10))
names = [meta[f"$P{i}S"] for i in range(3, 13)]
for i in range(1, 11):
    for j in range(1, 11):
        compensation_matrix[i - 1, j - 1] = meta[f"$DFC{i}TO{j}"]
print(compensation_matrix)

np.savetxt("/data/flowcat-data/paper-cytometry-resubmit/compensation_matrix_c0_t1.txt", compensation_matrix)

# TODO: check that voltages do not change in our dataset

# TODO: extract compensation tables for all samples and compare

for sample in case.samples:
    section("Tube " + sample.tube, level=5)
    data, meta = fcsparser.parse(data.complete_path)
    print(data)
