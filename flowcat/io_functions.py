from __future__ import annotations
from typing import Union
import json
import re
import logging
import pickle
from datetime import date

import joblib
import pandas as pd

from flowcat import mappings
from flowcat.utils.time_timers import str_to_date
from flowcat.utils.urlpath import URLPath
from flowcat.sommodels import fcssom
from flowcat.sommodels.casesom import CaseSingleSom, CaseSom
from flowcat.dataset import case, case_dataset, sample
from flowcat.dataset.som import SOM, SOMCollection


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
    with path.open("r") as jspath:
        data = json.load(jspath, object_hook=as_fc)
    return data


def save_json(data, path: URLPath):
    """Write json data to a file as a simple function."""
    with path.open("w") as jsfile:
        json.dump(data, jsfile, cls=FCEncoder)


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


def save_som(
        som: Union[SOMCollection, SOM],
        path: URLPath,
        subdirectory: bool = False,
        save_config: bool = True):
    """Save som object to the given destination.
    Params:
        som: Either a SOM collection or a single SOM object.
        path: Destination path
        subdirectory: Save files to separate files with path as directory name.
    """
    if isinstance(som, SOMCollection):
        soms = som
    elif isinstance(som, SOM):
        soms = [som]
    else:
        raise TypeError

    for som_obj in soms:
        data_dest, conf_dest = get_som_tube_path(path, som_obj.tube, subdirectory)
        LOGGER.debug("Saving %s to %s", som_obj, data_dest)
        save_csv(som_obj.data, data_dest)
        if save_config:
            save_json(som_obj.config, conf_dest)


def load_som(
        path: URLPath,
        subdirectory: bool = False,
        tube: Union[int, list] = None) -> Union[SOMCollection, SOM]:
    """Load soms into a som collection or if tube specified into a single SOM."""
    # Load single SOM
    if isinstance(tube, int):
        inpaths = get_som_tube_path(path, tube, subdirectory)
        return load_som(*inpaths, tube=tube)
    # Load multiple SOM into a SOM collection
    return load_som_collection(path, tubes=tube, subdirectory=subdirectory)


def load_single_som(data_path, config_path=None, **kwargs):
    data = load_csv(data_path)

    if config_path:
        try:
            config = load_json(config_path)
        except FileNotFoundError:
            config = {}
    else:
        config = {}

    kwargs = {**config, **kwargs}
    return SOM(data, **kwargs)


def load_som_collection(path, subdirectory: bool, tubes: list = None, **kwargs):
    path = URLPath(path)
    if tubes:
        tubepaths = {
            tube: get_som_tube_path(path, tube, subdirectory)[0] for tube in tubes
        }
    else:
        if subdirectory:
            paths = path.glob("t*.csv")
        else:
            parent = path.local.parent
            paths = [p for p in parent.glob(f"{path.local.name}*.csv")]

        tubepaths = {
            int(m[1]): p for m, p in
            [(re.search(r"t(\d+)\.csv", str(path)), path) for path in paths]
            if m is not None
        }
    tubes = sorted(tubepaths.keys())
    # load config if exists
    conf_path = path / "config.json"
    if conf_path.exists():
        config = load_json(conf_path)
    else:
        config = None
    return SOMCollection(path=path, tubes=tubes, tubepaths=tubepaths, config=config)


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
    save_som(model.weights, path / "weights", subdirectory=True)


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


def load_case_collection(data_path: URLPath, meta_path: URLPath):
    cases = load_json(meta_path)
    cases.data_path = data_path
    return cases
