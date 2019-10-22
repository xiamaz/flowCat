from flowcat import io_functions, utils
from flowcat.utils import time_timers

data = utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F")
meta1 = utils.URLPath("output/0-final/dataset/train.json.gz")
meta2 = utils.URLPath("output/0-final/dataset/test.json.gz")

with time_timers.timer("Test"):
    print("Hello")


with time_timers.timer("DS1"):
    dataset = io_functions.load_case_collection(data, meta1)

print(dataset)


with time_timers.timer("DS2"):
    test_dataset = io_functions.load_case_collection(data, meta2)

print(test_dataset)
