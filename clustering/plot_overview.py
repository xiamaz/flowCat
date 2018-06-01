import json
import matplotlib.pyplot as plt

with open("case_info.json") as casefile:
    data = json.load(casefile)

cohort_nums = {c: len(cc) for c, cc in data.items()}


fig = plt.figure()

plt.bar(cohort_nums.keys(), cohort_nums.values())

plt.xlabel("Kohorten")
plt.ylabel("Anzahl")
