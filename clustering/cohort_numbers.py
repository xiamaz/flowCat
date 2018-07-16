#!/usr/bin/env python3
# Small script to output usable numbers of cases after filtering for tubes and
# marker identification
#
# Will also return marker composition
from clustering.collection import CaseCollection

collection = CaseCollection("s3://mll-flowdata/case_info.json", [1, 2])

# simulate our clustering environment to get realistic numbers
all_view = collection.create_view()

print("Total: {}\n".format(len(all_view)))

print("Cohorts")
print("\n".join(
    ["{}: {}".format(k, len(v)) for k, v in all_view.groups.items()]
))

print("\n")

print("Materials")
for t in [1, 2]:
    tube_view = all_view.get_tube(t)
    print("Tube {}".format(t))
    print("\n".join([
        "{}: {}".format(k, len(v))
        for k, v in tube_view.materials.items()
    ]))
    print("Total: {}".format(len(tube_view)))
    print("\n")

    print("Channels: {}".format(len(tube_view.markers)))
    print("\n".join(tube_view.markers))
    print("\n")

dissimilar_tubes = []
for single in all_view:
    if len(set(
            [single.get_tube(p).material for p in [1, 2]]
    )) != 1:
        dissimilar_tubes.append(single)

print("Dissimilar: {}".format(len(dissimilar_tubes)))
print("\n".join([
    "{}|{}: {}".format(
        d.group, d.id, ", ".join([
            str(d.get_tube(p).material) for p in [1, 2]
        ])
    ) for d in dissimilar_tubes
]))
