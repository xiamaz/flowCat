from typing import Union
import json
import re
import gzip
import logging
import pickle
import shutil
from datetime import date

import joblib
import numpy as np
import pandas as pd

from flowcat import mappings
from flowcat.utils.time_timers import str_to_date
from flowcat.utils.urlpath import URLPath
from flowcat.sommodels import fcssom
from flowcat.sommodels.casesom import CaseSingleSom, CaseSom
from flowcat.dataset import case, case_dataset, sample
from flowcat.dataset.som import SOM


LOGGER = logging.getLogger(__name__)


class FCEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=E0202
        if type(obj) in mappings.PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        elif isinstance(obj, URLPath):
            return {"__urlpath__": str(obj)}
        elif isinstance(obj, case_dataset.CaseCollection):
            return {
                "__casecollection__": case_dataset.case_collection_to_json(obj)
            }
        elif isinstance(obj, case.Case):
            return {
                "__case__": case.case_to_json(obj)
            }
        elif isinstance(obj, sample.FCSSample):
            return {
                "__fcssample__": sample.json_to_fcssample(obj)
            }
        elif isinstance(obj, sample.SOMSample):
            return {
                "__somsample__": sample.json_to_somsample(obj)
            }
        elif isinstance(obj, date):
            return {
                "__date__": obj.isoformat()
            }
        return json.JSONEncoder.default(self, obj)


def as_fc(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(mappings.PUBLIC_ENUMS[name], member)
    elif "__urlpath__" in d:
        return URLPath(d["__urlpath__"])
    elif "__casecollection__" in d:
        return case_dataset.json_to_case_collection(d["__casecollection__"])
    elif "__case__" in d:
        return case.json_to_case(d["__case__"])
    elif "__fcssample__" in d:
        return sample.json_to_fcssample(d["__fcssample__"])
    elif "__somsample__" in d:
        return sample.json_to_somsample(d["__somsample__"])
    elif "__date__" in d:
        return str_to_date(d["__date__"])
    else:
        return d


def load_json(path: URLPath):
    """Load json data from a path as a simple function."""
    if path.suffix == ".gz":
        jsfile = gzip.open(str(path), "r", compresslevel=1)
    else:
        jsfile = path.open("r")
    data = json.load(jsfile, object_hook=as_fc)

    jsfile.close()
    return data


def save_json(data, path: URLPath):
    """Write json data to a file as a simple function."""
    if path.suffix == ".gz":
        jsfile = gzip.open(str(path), "wt", compresslevel=1)
    else:
        jsfile = path.open("w")

    json.dump(data, jsfile, cls=FCEncoder)

    jsfile.close()


def load_pickle(path: URLPath):
    with path.open("rb") as pfile:
        data = pickle.load(pfile)
    return data


def save_pickle(data, path: URLPath):
    """Write data to the given path as a pickle."""
    with path.open("wb") as pfile:
        pickle.dump(data, pfile)


def load_joblib(path: URLPath):
    return joblib.load(str(path))


def save_joblib(data, path: URLPath):
    path.parent.mkdir()
    with path.open("wb") as handle:
        joblib.dump(data, handle)


def to_json(data):
    return json.dumps(data, indent=4)


def load_csv(path, index_col=0):
    data = pd.read_csv(str(path), index_col=index_col)
    return data


def save_csv(data: pd.DataFrame, path: URLPath):
    path.parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(path)


def get_som_tube_path(
        path: URLPath,
        tube: str,
        subdirectory: bool) -> URLPath:
    if subdirectory:
        result = path / f"t{tube}"
    else:
        result = path + f"_t{tube}"
    return (result + ".csv", result + ".json")


def save_som(som: SOM, path: URLPath, save_config: bool = True):
    npy_path = path.with_suffix(".npy")
    np.save(str(npy_path), som.data)

    if save_config:
        meta_path = path.with_suffix(".json")
        save_json({"markers": som.markers}, meta_path)


def load_som(path: URLPath, load_config: bool = True) -> SOM:
    if load_config:
        meta_path = path.with_suffix(".json")
        config = load_json(meta_path)
    else:
        config = {}

    npy_path = path.with_suffix(".npy")
    return SOM(data=npy_path, **config)


def load_casesom(path: URLPath, tensorboard_dir: URLPath = None, **kwargs):
    singlepaths = {p.name.lstrip("tube"): p for p in path.iterdir() if "tube" in str(p)}
    models = {}
    for tube, mpath in sorted(singlepaths.items()):
        tbdir = tensorboard_dir / f"tube{tube}" if tensorboard_dir else None
        models[tube] = load_casesinglesom(mpath, tensorboard_dir=tbdir, **kwargs)
    return CaseSom(models=models)


def save_casesom(model, path: URLPath):
    for tube, tmodel in model.models.items():
        output_path = path / f"tube{tube}"
        save_casesinglesom(tmodel, output_path)


def load_casesinglesom(path: URLPath, **kwargs):
    config = load_json(path / "casesinglesom_config.json")
    model = load_fcssom(path, **kwargs)
    return CaseSingleSom(model=model, **config)


def save_casesinglesom(model, path: URLPath):
    save_fcssom(model.model, path)
    save_json(model.config, path / "casesinglesom_config.json")
    save_som(model.weights, path / f"weights", save_config=True)


def load_fcssom(path: URLPath, **kwargs):
    scaler = load_joblib(path / "scaler.joblib")
    config = load_json(path / "config.json")
    model = fcssom.FCSSom(
        dims=config["dims"],
        scaler=scaler,
        name=config["name"],
        markers=config["markers"],
        marker_name_only=config["marker_name_only"],
        **{**config["modelargs"]["kwargs"], **kwargs},
    )
    model.model.load(path / "model.ckpt")
    model.trained = True
    return model


def save_fcssom(model, path: URLPath):
    if not model.trained:
        raise RuntimeError("Model has not been trained")

    path.mkdir(parents=True, exist_ok=True)
    model.model.save(path / "model.ckpt")
    save_joblib(model.scaler, path / "scaler.joblib")
    save_json(model.config, path / "config.json")


def load_case_collection_from_caseinfo(data_path: URLPath, meta_path: URLPath):
    """Load case collection from caseinfo json, as used in the MLL dataset."""
    metadata = load_json(meta_path)
    metaconfig = {}
    data = [case.caseinfo_to_case(d, data_path) for d in metadata]

    metaconfig["data_path"] = data_path
    metaconfig["meta_path"] = meta_path

    return case_dataset.CaseCollection(data, **metaconfig)


def save_case_collection(cases, destination: URLPath):
    save_json(cases, destination)


def save_case_collection_with_data(cases, destination: URLPath):
    sample_destination = destination / "data"
    cases = cases.copy()
    for case_obj in cases:
        for case_sample in case_obj.samples:
            sdest = sample_destination / case_sample.path
            sdest.parent.mkdir()
            shutil.copy(str(case_sample.complete_path), sdest)
            case_sample.dataset_path = sample_destination

    save_case_collection(cases, destination=destination / "meta.json.gz")
    return cases


def load_case_collection(data_path: URLPath, meta_path: URLPath):
    cases = load_json(meta_path)
    if not isinstance(cases, case_dataset.CaseCollection):
        raise TypeError("Loaded json does not contain valid case collection.")
    cases.set_data_path(data_path)
    return cases
