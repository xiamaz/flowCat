import json
import os
import shutil
import random
import collections
import flowcat

cases = flowcat.CaseCollection.from_path("/data/flowcat-data/mll-flowdata/decCLL-9F")

def move_cases(cases, output_path):
    os.makedirs(output_path)
    for case in cases:
        print(case.id)
        for filepath in case.filepaths:
            srcpath = filepath.localpath
            filename = f"{case.id}_t{filepath.tube}.fcs"
            filepath.path = filename
            dstpath = f"{output_path}/{filename}"
            print("Copying ", srcpath, dstpath)
            shutil.copy(srcpath, dstpath)
        case.path = output_path
        print("----")

    caseinfo = [c.json for c in cases]
    with open(f"{output_path}/case_info.json", "w") as fp:
        json.dump(caseinfo, fp)

def move_subsample(output_path):
    groups = collections.defaultdict(list)
    for case in cases:
        groups[case.group].append(case)

    stratified = [case for cases in groups.values() for case in random.sample(cases, 20)]
    counts = collections.Counter(s.group for s in stratified)
    print(counts)
    move_cases(stratified, output_path)


def move_missing(output):
    selected = {1: ['CD14-APCA750']}
    result = cases.search(selected)
    move_cases(result, output)


if __name__ == "__main__":
    # move_subsample("output/subsample")
    move_missing("output/missing")
