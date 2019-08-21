"""Create case collection from berlin json dataset.

Creates a new casecollection json in the lb-data2 folder, which we can
thereafter directly load using casecollection load.
"""
# pylint: skip-file
# flake8: noqa
import json
from datetime import datetime

import flowcat
from flowcat.utils import URLPath


def convert_timestamp(timestamp: str) -> str:
    """Create datetime object from the given timestamp"""
    return datetime.fromisoformat(timestamp).date().isoformat()


def meta_to_filepath(metadata: dict, data_url: "URLPath", tube: str) -> dict:
    """Create filepath dictionary."""
    fcs_path = data_url / f"c{metadata['id']}_t{tube}.lmd"
    filepath = {
        "fcs": {
            "path": fcs_path
        },
        "date": convert_timestamp(metadata["date"]),
        "tube": tube,
        "material": "KM",
        "panel": "B-NHL",
    }
    return filepath


def meta_to_case(metadata: dict, data_url: "URLPath") -> flowcat.dataset.case.Case:
    """Generate case objects from the given metadata dict."""
    filepaths = [meta_to_filepath(metadata, data_url, str(tube)) for tube in metadata["tubes"]]

    casedict = {
        "id": str(metadata["id"]),
        "date": convert_timestamp(metadata["date"]),
        "cohort": metadata["class"],
        "diagnosis": metadata["diag_text"],
        "filepaths": filepaths
    }

    case = flowcat.dataset.case.Case(data=casedict, path=data_url)
    case.set_fcs_info()
    return case


dataset_path = URLPath("/data/flowcat-data/lb-data2")

fcs_data_path = dataset_path / "data"

with (dataset_path / "metadata.json").open("r") as metafile:
    metadata = json.load(metafile)

print(metadata[0])

data = [meta_to_case(meta, fcs_data_path) for meta in metadata]

dataset = flowcat.CaseCollection(data, path=fcs_data_path)

outpath = flowcat.utils.URLPath("output/50-berlin-data/dataset")
dataset.save(dataset_path / "casecollection")

# only use groups we already have for now
group_dataset, reasons = dataset.filter_reasons(groups=flowcat.mappings.GROUPS)
group_dataset.save(dataset_path / "valid_groups")

invalid_labels = [l for l, _ in reasons]
invalid_dataset, _ = dataset.filter_reasons(labels=invalid_labels)
invalid_dataset.save(dataset_path / "invalid_groups")

selected = flowcat.marker_selection.get_selected_markers(
    group_dataset,
    ("1", "2", "3", "4"),
    marker_threshold=0.9)
selected.save(outpath / "dataset" / "known_groups")

selected_invalid, _ = invalid_dataset.filter_reasons(selected_markers=selected.selected_markers)
selected_invalid.save(outpath / "dataset" / "unknown_groups")
