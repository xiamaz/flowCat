# pylint: skip-file
# flake8: noqa
import numpy as np
import pandas as pd

from flowcat.utils import URLPath
from flowcat.io_functions import load_case_collection
from flowcat.dataset import case_dataset


datapath = URLPath("output/test-2019-08/som")
metapath = URLPath("output/test-2019-08/som.json")
cases = load_case_collection(datapath, metapath)
print(cases)


nppath = URLPath("output/test-2019-08/somnp2")
nppath.mkdir()
all_num = len(cases)
for i, case in enumerate(cases):
    print(f"Converting {i}/{all_num}")
    for somsample in case.samples:
        somsample.path = datapath / f"{case.id}_t{somsample.tube}.csv"
        somdata = pd.read_csv(str(somsample.path), index_col=0)
        somarray = somdata.values
        somarray = somarray.reshape((32, 32, -1))
        newpath = nppath / f"{case.id}_t{somsample.tube}.npy"
        np.save(str(newpath), somarray)
