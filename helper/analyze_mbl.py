import os
import csv
from functools import reduce
from collections import defaultdict


def update_dict(x, y):
    label, length = y
    x[length].append(label)
    return x


names = ["label", "date", "patient", "infiltration"]

with open("MBL.csv") as mblfile:
    dicts = csv.DictReader(mblfile, fieldnames=names, delimiter=";")
    examinations = {d['label']: d for d in dicts}

mbl_files = list(os.listdir("./MBL"))

not_contained = [m for m in mbl_files if not
                 any([l in m for l in examinations])]
print("Not contained in any file")
print(not_contained)

examinations = {
    l: {"info": i, "files": [f for f in mbl_files if l in f and ".LMD" in f]}
    for l, i in examinations.items()
}

filelens = {l: len(v['files']) for l, v in examinations.items()}

r = reduce(update_dict, filelens.items(), defaultdict(list))
print("Examination IDs without any files.")
print(r[0])
