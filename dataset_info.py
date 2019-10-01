from flowcat import io_functions, utils

data = utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F")
meta = utils.URLPath("output/0-munich-data/dataset/train.json")
meta_test = utils.URLPath("output/0-munich-data/dataset/test.json")

dataset = io_functions.load_case_collection(data, meta)
print(dataset)

test_dataset = io_functions.load_case_collection(data, meta_test)
print(test_dataset)
