from flowcat import io_functions, utils
from flowcat.utils import time_timers

data = utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F")
meta1 = utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F/case_info_2018-12-15.json")
meta2 = utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F/case_info_diagnosis_2018-12-15.json.gz")

with time_timers.timer("Test"):
    print("Hello")


with time_timers.timer("DS1"):
    dataset = io_functions.load_case_collection_from_caseinfo(data, meta1)

print(dataset)


with time_timers.timer("DS2"):
    dataset = io_functions.load_case_collection_from_caseinfo(data, meta2)

print(dataset)
