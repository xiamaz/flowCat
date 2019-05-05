import shutil
import random
import collections
import flowcat

output_path = "output/subsample"

cases = flowcat.CaseCollection.from_path("/data/flowcat-data/mll-flowdata/decCLL-9F")

groups = collections.defaultdict(list)
for case in cases:
    groups[case.group].append(case)

stratified = [case for cases in groups.values() for case in random.sample(cases, 20)]
counts = collections.Counter(s.group for s in stratified)
print(counts)

for case in stratified:
    print(case.id)
    for filepath in case.filepaths:
        srcpath = filepath.localpath
        dstpath = f"{output_path}/{case.id}_t{filepath.tube}.fcs"
        print(srcpath, dstpath)
    print("----")
