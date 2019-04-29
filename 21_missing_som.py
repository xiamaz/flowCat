# Sanely handle missing values in SOM
import flowcat
from flowcat.models import tfsom

cases = flowcat.CaseCollection.from_path("/data/flowcat-data/mll-flowdata/decCLL-9F")

# non_full = {}
# for tube in cases.tubes:
#     markers = cases.get_markers(tube)
#     non_full[tube] = [m for m, v in markers.items() if v < 1.0]

# select sample with missing tubes
selected = {1: ['CD14-APCA750']}
result = cases.search(selected)

# select reference som which doesnt have it
ref = flowcat.SOMCollection.from_path("output/mll-sommaps/reference_maps/CLL_i10")
data = ref.get_tube(1)
print(data)
print(data.markers)

model = flowcat.FCSSomTransformer(
    dims=(-1, -1, -1), init="reference", init_data=data,
    max_epochs=2,
    batch_size=1024,
    initial_radius=4, end_radius=1, radius_cooling="linear",
    tensorboard_dir="output/tensorboard",
)

fcssamples = [r.get_tube(1).data for r in result]
model.train(fcssamples)

print(model.weights)
#print(model)
