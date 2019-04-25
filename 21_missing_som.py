# Sanely handle missing values in SOM
import flowcat

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
