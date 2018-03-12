'''
Scramble data labels for testing usage.
---
Only use this for upsampling data.
'''
import os
import csv
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()
with open(args.filename, "r") as inputfile:
    result_dict = list(csv.DictReader(inputfile, quotechar="\"",
                                      delimiter=";"))

random_groups = {}
result_dicts = []
for i, item in enumerate(result_dict):
    headers = list(item.keys())
    for key, value in item.items():
        if key == "label":
            item[key] = "anon{}".format(i)
        elif key == "group":
            group = item[key]
            if group not in random_groups:
                random_groups[group] = "group{}".format(i)
            item[key] = random_groups[group]
    result_dicts.append(item)

with open("{}_scrambled{}".format(
        *os.path.splitext(args.filename)), "w") as scrambled:
    writer = csv.DictWriter(scrambled, fieldnames=headers, quotechar="\"",
                            delimiter=";")
    writer.writeheader()
    writer.writerows(result_dict)
