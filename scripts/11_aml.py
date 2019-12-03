from flowcat.utils import save_json
from flowcat.dataset.case_dataset import CaseCollection


def csv_to_case_dict(data):
    for cid, gdata in data.groupby("Individual"):
        tubes = gdata["Tube number"]
        first = gdata.iloc[0]
        filepaths = [
            {
                "fcs": {"path": f},
                "date": "2012-04-04",
                "tube": tubes.iloc[i],
                "material": "KM",
            }
            for i, f in enumerate(gdata["FCS file"])
        ]
        cdict = {
            "id": str(first["Individual"]),
            "group": first["Condition"],
            "filepaths": filepaths,
            "date": "2012-04-04",
        }
        yield cdict


cases = CaseCollection.from_path("/data/flowcat-data/flowcap-aml", metapath="/data/flowcat-data/flowcap-aml/attachments/AML.csv", transfun=csv_to_case_dict)
cases.set_markers()

save_json(cases.json, "/data/flowcat-data/flowcap-aml/case_info.json")
