# Sanely handle missing values in SOM
import logging
import sklearn as sk
import flowcat
from flowcat.models import tfsom


def configure_print_logging(rootname="flowcat"):
    """Configure default logging for visual output to stdout."""
    handlers = [
        flowcat.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    flowcat.add_logger(rootname, handlers, level=logging.DEBUG)


def train_native():
    ds_missing = flowcat.CaseCollection.from_path("output/missing")
    ds_subsample = flowcat.CaseCollection.from_path("output/subsample")

    tube = 1
    markers = ds_subsample.get_markers(tube)
    markers_list = markers.index.values

    filepath = ds_missing.data[0].get_tube(tube)
    fcsdata = filepath.data
    fcsdata = fcsdata.align(markers_list)

    transformer = sk.preprocessing.MinMaxScaler()
    data = transformer.fit_transform(fcsdata.data)

    model = tfsom.TFSom(32, 32, markers_list)
    model.train(data)


# select reference som which doesnt have it
# ref = flowcat.SOMCollection.from_path("output/mll-sommaps/reference_maps/CLL_i10")
# data = ref.get_tube(1)
#
# model = flowcat.FCSSomTransformer(
#     dims=(-1, -1, -1), init="reference", init_data=data,
#     max_epochs=2,
#     batch_size=1024,
#     initial_radius=4, end_radius=1, radius_cooling="linear",
#     tensorboard_dir="output/tensorboard",
# )
# 
# fcssamples = [r.get_tube(1).data for r in result]
# model.train(fcssamples)
# 
# print(model.weights)
#print(model)

if __name__ == "__main__":
    configure_print_logging()
    train_native()
