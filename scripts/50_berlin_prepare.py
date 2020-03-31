"""Create case collection from berlin json dataset.

Creates a new casecollection json in the lb-data2 folder, which we can
thereafter directly load using casecollection load.
"""
# pylint: skip-file
# flake8: noqa
import json
from datetime import datetime

from flowcat.types.material import Material
from flowcat.constants import GROUPS
from flowcat.dataset import case as fc_case, sample as fc_sample
from flowcat.dataset.case_dataset import CaseCollection
from flowcat.io_functions import save_case_collection, load_case_collection, save_json
from flowcat.utils import URLPath


def convert_timestamp(timestamp: str) -> str:
    """Create datetime object from the given timestamp"""
    return datetime.fromisoformat(timestamp).date()


def meta_to_filepath(metadata: dict, data_url: "URLPath", tube: str, case_id: str) -> dict:
    """Create filepath dictionary."""
    fcs_path = data_url / f"c{metadata['id']}_t{tube}.lmd"
    date = convert_timestamp(metadata["date"])
    sample_id = f"{case_id}_t{tube}_{date.isoformat()}"
    filepath = {
        "path": fcs_path,
        "id": sample_id,
        "case_id": case_id,
        "date": date,
        "tube": tube,
        "panel": "B-NHL",
        "material": Material.PERIPHERAL_BLOOD,
    }

    sample = fc_sample.FCSSample(**filepath)
    return sample


def meta_to_case(metadata: dict, data_url: "URLPath") -> fc_case.Case:
    """Generate case objects from the given metadata dict."""
    case_id = str(metadata["id"])
    filepaths = [meta_to_filepath(metadata, data_url, str(tube), case_id) for tube in metadata["tubes"]]

    casedict = {
        "id": case_id,
        "date": convert_timestamp(metadata["date"]),
        "group": metadata["class"],
        "diagnosis": metadata["diag_text"],
        "samples": filepaths
    }

    case = fc_case.Case(**casedict)
    case.set_fcs_info()
    print(case)
    return case


dataset_path = URLPath("/data/flowcat-data/lb-data2")

fcs_data_path = dataset_path / "data"

with (dataset_path / "metadata.json").open("r") as metafile:
    metadata = json.load(metafile)

print(metadata[0])

data = [meta_to_case(meta, fcs_data_path) for meta in metadata]

dataset = CaseCollection(data, data_path=fcs_data_path)

outpath = URLPath("output/5-berlin-data-test/dataset")

save_case_collection(dataset, dataset_path / "casecollection.json")

# only use groups we already have for now
group_dataset, reasons = dataset.filter_reasons(groups=GROUPS)
save_case_collection(group_dataset, dataset_path / "valid_groups.json")

invalid_labels = [l for l, _ in reasons]
invalid_dataset, _ = dataset.filter_reasons(labels=invalid_labels)
save_case_collection(invalid_dataset, dataset_path / "invalid_groups.json")

selected = flowcat.marker_selection.get_selected_markers(
    group_dataset,
    ("1", "2", "3", "4"),
    marker_threshold=0.9)
save_case_collection(selected, outpath / "known_groups.json")

references = selected.sample(num=1)
save_json(references, output / "references.json")

selected_invalid, _ = invalid_dataset.filter_reasons(selected_markers=selected.selected_markers)
save_case_collection(selected_invalid, outpath / "unknown_groups.json")
