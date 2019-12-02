from flowcat import flowcat


model = flowcat.FlowCat.load(
    ref_path="/data/flowcat-data/paper-cytometry/reference",
    cls_path="/data/flowcat-data/paper-cytometry/classifier")


dataset = flowcat.load_case_collection(
    "/data/flowcat-data/paper-cytometry/unused-data"
)
print(dataset)
print(dataset.group_count)

testcase = dataset[0]

pred = model.predict(testcase)

grads = model.generate_saliency(pred, pred.predicted_group)
print(grads)
