from flowcat import io_functions, utils
from flowcat.utils import time_timers

data = utils.URLPath("/data/flowcat-data/2019-10_paper_data/unknown-cohorts")
meta = utils.URLPath("/data/flowcat-data/2019-10_paper_data/unknown-cohorts/meta.json")

with time_timers.timer("DS1"):
    dataset = io_functions.load_case_collection(data, meta)

print(dataset)
