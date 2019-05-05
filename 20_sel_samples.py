import json
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
        filename = f"{case.id}_t{filepath.tube}.fcs"
        filepath.path = filename
        dstpath = f"{output_path}/{filename}"
        print("Copying ", srcpath, dstpath)
        shutil.copy(srcpath, dstpath)
    case.path = output_path
    print("----")

caseinfo = [c.json for c in stratified]
with open(f"{output_path}/case_info.json", "w") as fp:
    json.dump(caseinfo, fp)
