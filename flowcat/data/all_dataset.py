"""
Dataset combining information from FCS and other transformed sources.
"""
from . import case_dataset, loaders


class HistoDataset:
    """Information from histogram distribution experiments."""

    def __init__(self, path):
        """Path to histogram dataset. Should contain a dataframe for
        each available tube."""
        self._data = self.read_path(path)

    @staticmethod
    def read_path(path):
        """Read the given path and return a label mapped to either the actual
        data or a path."""


class SOMDataset:
    """Infomation in self-organizing maps."""

    def __init__(self, path):
        """Path to SOM dataset. Should have another csv file with metainfo
        and individual SOM data inside the directory."""
        pass


class CombinedDataset:
    """Combines information from different data sources."""
    pass


def rescale_sureness(data):
    data["sureness"] = data["sureness"] / data["sureness"].mean() * 5
    return data


def create_dataset(tubes, som=None, histo=None, fcs=None, min_fcs_count=None):
    """Creata a new dataset table containing path information to different
    path sources.
    Args:
        tubes: Tubes to be used to compile information.
        som: Path to directory containing a SOM dataset
        histo: Path to directory containing a histo dataset
        fcs: Path to directory containing FCS dataset
        min_fcs_count: Number of FCS events necessary for inclusion.
    Returns:
        Dataframe contaning path information for different data sources.
    """
    all_counts = {}
    if som is not None:
        mappath = utils.URLPath(som)
        sommap_labels = utils.load_csv(mappath + ".csv")
        sommap_labels.set_index(["label", "group"], inplace=True)
        sommap_count = pd.DataFrame(1, index=sommap_labels.index, columns=["count"])
        all_counts["som"] = sommap_count
    if histo is not None:
        histopath = utils.URLPath(histo)
        sum_count = None
        for tube in tubes:
            df = LoaderMixin.read_histo_df(
                histopath / f"tube{tube}.csv")
            count = pd.DataFrame(1, index=df.index, columns=["count"])
            if sum_count is None:
                sum_count = count
            else:
                sum_count.add(count, fill_value=0)
        histo_count = sum_count == len(tubes)
        all_counts["histo"] = histo_count

    cdict = {}
    cases = cc.CaseCollection(fcs, tubes=[1, 2])
    caseview = cases.filter(counts=10000)
    for case in caseview:
        material = case.has_same_material([1, 2])
        fcspaths = {t: str(case.get_tube(t, material=material, min_count=10000).path.local) for t in [1, 2]}
        try:
            assert both_count.loc[(case.id, case.group), "count"] == 3, "Not all data available."
            cdict[case.id] = {
                "group": case.group,
                "sommappath": str(mappath / f"{case.id}_t{{tube}}.csv"),
                "fcspath": fcspaths,
                "histopath": f"{histo}/tube{{tube}}.csv",
                "sureness": case.sureness,
            }
        except KeyError as e:
            LOGGER.debug(f"{e} - Not found in histo or sommap")
            continue
        except AssertionError as e:
            LOGGER.debug(f"{case.id}|{case.group} - {e}")
            continue

    dataset = pd.DataFrame.from_dict(cdict, orient="index")

    # scale sureness to mean 5 per group
    dataset = dataset.groupby("group").apply(rescale_sureness)
    return dataset


def load_dataset(index=None, paths=None, mapping=None):
    """Return dataframe containing columns with filename and labels.
    Args:
    """
    if index is None:
        dataset = create_dataset(**paths)
    else:
        dataset = utils.load_pickle(utils.URLPath(index))

    mapdict = GROUP_MAPS[mapping]
    dataset = dataset_apply_mapping(dataset, mapdict)
    return dataset, mapdict


def modify_groups(data, mapping):
    """Change the cohort composition according to the given
    cohort composition."""
    data["group"] = data["group"].apply(lambda g: mapping.get(g, g))
    return data


def dataset_apply_mapping(dataset, mapping):
    """Apply a specific mapping to the given dataset."""
    # copy group into another column
    dataset["orig_group"] = dataset["group"]
    if mapping is not None:
        dataset = modify_groups(dataset, mapping=mapping["map"])
        dataset = dataset.loc[dataset["group"].isin(mapping["groups"]), :]
    return dataset


def split_dataset(data, test_num=None, test_labels=None, train_labels=None):
    """Split data in stratified fashion by group.
    Args:
        data: Dataset to be split. Label should be contained in 'group' column.
        test_num: Ratio of samples in test per group or absolute number of samples in each group for test.
    Returns:
        (train, test) with same columns as input data.
    """
    if test_labels is not None:
        if not isinstance(test_labels, list):
            test_labels = utils.load_json(test_labels)
        test = data.loc[test_labels, :]
    if train_labels is not None:
        if not isinstance(train_labels, list):
            train_labels = utils.load_json(train_labels)
        train = data.loc[train_labels, :]
    if test_num is not None:
        assert test_labels is None and train_labels is None, "Cannot use num with specified labels"
        grouped = data.groupby("group")
        if test_num < 1:
            test = grouped.apply(lambda d: d.sample(frac=test_num)).reset_index(level=0, drop=True)
        else:
            group_sizes = grouped.size()
            if any(group_sizes <= test_num):
                insuff = group_sizes[group_sizes <= test_num]
                LOGGER.warning("Insufficient sizes: %s", insuff)
                raise RuntimeError("Some cohorts are too small.")
            test = grouped.apply(lambda d: d.sample(n=test_num)).reset_index(level=0, drop=True)
        train = data.drop(test.index, axis=0)
    return train, test
