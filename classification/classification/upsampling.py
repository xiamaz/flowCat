'''
CSV file operations needed for upsampling based classification
'''
import re
import logging
from enum import Enum
from functools import reduce
import typing

import pandas as pd


class ViewModifiers(Enum):
    SMALLEST = 1

    @staticmethod
    def from_modifiers(modifiers: list) -> list:
        return [ViewModifiers[m.upper()] for m in modifiers]


SizesDict = typing.Dict[str, int]
FilesDict = typing.Dict[int, typing.Dict[int, str]]
GroupSelection = typing.List[str]
SizeOption = typing.Union[int, SizesDict]
MaybeList = typing.Union[typing.List[int], None]


# Groups without infiltration, if using this option for splitting the data
NO_INFILTRATION = ["normal"]


RE_TUBE = re.compile(r"tube(\d+)\.csv")

LOGGER = logging.getLogger(__name__)


def merge_on_label(left_df: pd.DataFrame, right_df: pd.DataFrame, metacols) \
        -> pd.DataFrame:
    '''Merge two dataframes on label column and drop additional replicated
    columns such as group.'''
    merged = left_df.merge(
        right_df, how="inner", on="label", suffixes=("", "_y")
    )
    merged.drop([
        s + "_y" for s in metacols if s != "label"
    ], inplace=True, axis=1)

    # continuous field naming
    names = [
        m for m in list(merged)
        if m not in metacols
    ]
    rename_dict = {m: str(i + 1) for i, m in enumerate(names)}
    merged.rename(columns=rename_dict, inplace=True)

    # reorder the columns
    merged_names = list(rename_dict.values()) + metacols
    merged = merged[merged_names]
    return merged


class BaseData:
    """Base data class containing filtering and basic abstractions."""
    def __init__(self, data: pd.DataFrame, name: str = ""):
        self._data = None
        self._groups = None

        self.data = data
        self.name = name

    @classmethod
    def from_obj(cls, obj):
        return cls(obj.data, obj.name)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._groups = self.split_groups(self._data)

    @property
    def group_names(self):
        return [n for n, _ in self._groups]

    @property
    def group_sizes(self):
        return {n: len(d) for n, d in self._groups}

    @staticmethod
    def split_groups(data):
        '''Split data into a list of group_name, dataframe tuples.
        '''
        return list(data.groupby("group"))

    def _get_group(self, group_info: dict) -> pd.DataFrame:
        """Select data according to group dict."""
        # get all data contained in the tags
        sel_data = self.data.loc[
            self.data["group"].isin(group_info["tags"])
        ].copy()

        # explicitly ok modifications of selection
        # rename group column in these data to the specified label
        sel_data.loc[:, "group"] = group_info["name"]
        return sel_data

    def select_groups(self, groups: [dict], sizes: dict) -> list:
        '''Select only certain groups.
        Input is a list of dicts, containing
        {
            "name": label of group,
            "tags": raw cohort labels to be assigned to this group
        }
        '''
        dfs = []
        for group in groups:
            groupdf = self._get_group(group)
            gname = group["name"]

            ginfo = sizes.get(gname, sizes[""])

            # exclude cohort if smaller than min cutoff
            if ginfo["min_size"] and groupdf.shape[0] < ginfo["min_size"]:
                groupdf = groupdf.iloc[0:0, :]
            # randomly downsample cohort if larger than max size
            elif ginfo["max_size"] \
                    and groupdf.shape[0] > ginfo["max_size"]:
                groupdf = groupdf.sample(n=ginfo["max_size"])

            dfs.append(groupdf)

        return dfs

    def filter_data(
            self,
            groups: list,
            sizes: dict,
            modifiers: [ViewModifiers]
    ) -> "BaseView":
        '''Filter data based on criteria and return a data view.'''
        if not groups:
            groups = [{"name": g, "tags": [g]} for g in self.group_names]

        df_list = self.select_groups(groups, sizes)

        for modifier in modifiers:
            if modifier == ViewModifiers.SMALLEST:
                smallest_group = min(map(lambda d: d.shape[0], df_list))

                df_list = [d.sample(n=smallest_group) for d in df_list]

        data = pd.concat(df_list)
        return self.__class__(data, name=self.name)

    @staticmethod
    def split_data_labels(
            dataframe: pd.DataFrame,
    ) -> (pd.DataFrame, pd.DataFrame):
        '''Split dataframe into matrices with group labels as sparse matrix'''
        data = dataframe.drop(
            [c for c in dataframe.columns if not c.isdigit()], axis=1
        )
        labels = dataframe['group']
        return data, labels


class DataView(BaseData):
    '''Contains upsampling data with possiblity to apply filters, keeping
    state.
    '''

    def get_test_train_split(
            self, ratio: float = None, abs_num: int = None,
            train_infiltration: float = 0.0
    ) -> (pd.DataFrame, pd.DataFrame):
        '''Test and training split for data evaluation.
        Ratio and abs_num are applied per cohort (eg unique label).

        Args:
            ratio: Relative ratio of train-test split
            abs: Absolute number in the train cohort
            train_infiltration: Minimum infiltration requirement for train
                inclusion
        Returns:
            Tuple of (train, test). Rows in each dataframe will be randomly
            shuffled.
        '''
        if not (ratio or abs_num):
            raise TypeError("Neither ratio nor absolute number cutoff given")

        test_list = []
        train_list = []
        for group_name, group in self._groups:
            LOGGER.info("Splitting %s", group_name)
            i = (group.shape[0] - abs_num) \
                if abs_num else int(ratio * group.shape[0])
            if i < 0:
                raise RuntimeError("Cutoff below zero. Perhaps small cohort.")
            shuffled = group.sample(frac=1)
            if train_infiltration and group_name not in NO_INFILTRATION:
                over_thres = shuffled["infiltration"].astype("float32") \
                    > train_infiltration
                hi_infil = shuffled.loc[over_thres]
                lo_infil = shuffled.loc[~over_thres]

                delta_train = max(i - hi_infil.shape[0], 0)
                if delta_train > 0:
                    LOGGER.warning((
                        "%s high infiltration only %d cases, "
                        "below cutoff %d required for given split."
                    ), group_name, hi_infil.shape[0], i)

                train = pd.concat([
                    hi_infil.iloc[:min(i, hi_infil.shape[0]), :],
                    lo_infil.iloc[:delta_train, :],
                ])
                test = pd.concat([
                    hi_infil.iloc[min(i, hi_infil.shape[0]):, :],
                    lo_infil.iloc[delta_train:, :],
                ])
            else:
                train, test = shuffled.iloc[:i, :], shuffled.iloc[i:, :]

            test_list.append(test)
            train_list.append(train)

        df_test = pd.concat(test_list)
        df_test = df_test.sample(frac=1)
        df_train = pd.concat(train_list)
        df_train = df_train.sample(frac=1)
        return df_train, df_test

    def k_fold_split(self, k_num: int = 5) -> [pd.DataFrame]:
        '''Split file into k same-size partitions. Keep the group ratios
        same.
        '''
        df_list = []
        for _, group in self._groups:
            shuffled = group.sample(frac=1)
            df_group = []
            block_size = round(group.shape[0] / k_num)
            for i in range(k_num):
                start = block_size * i
                end = block_size * (i + 1) if i + 1 < k_num else group.shape[0]
                group_slice = shuffled.iloc[start:end, :]
                df_group.append(group_slice)
            df_list.append(df_group)
        df_splits = [pd.concat(dfs) for dfs in zip(*df_list)]
        df_splits = [df.sample(frac=1) for df in df_splits]
        return df_splits


class InputData(BaseData):
    '''Data from upsampling in pandas dataframe form.

    This class handles parsing from the raw data format into a
    continuous pandas dataframe usable for later usage.
    '''

    old_format = {
        "sep": ";",
        "index_col": 0,
        "dtype": {
            "infiltration": object,
            "group": object,
            "label": object,
        }
    }

    new_format = {
        "sep": ",",
        "index_col": 0,
        "dtype": {
            "infiltration": "float32",
            "group": object,
            "label": object,
        },
    }

    @classmethod
    def from_files(
            cls, tubesdict: dict, name: str, tubes: MaybeList = None
    ) -> "UpsamplingData":
        '''Create upsampling data from structure containing multiple tubes
        for multiple files.
        The input structure contains a list of items to be joined by row,
        which are in turn joined on columns. (eg multiple tubes on the inner
        level split into multiple output files on the outer level)
        '''
        return cls(cls._merge_tubes(tubesdict, tubes), name)

    @staticmethod
    def _read_file(filepath: str, csv_format: str = "new") -> pd.DataFrame:
        '''Read output from preprocessing in csv format.
        '''
        if csv_format == "old":
            csv_data = pd.read_table(
                filepath,
                **InputData.old_format
            )
            csv_data.loc[:, "infiltration"] = csv_data["infiltration"].apply(
                lambda x: str(x).replace(",", ".")
            )
            csv_data.loc[:, "infiltration"] = csv_data["infiltration"].astype(
                "float32"
            )
        elif csv_format == "new":
            csv_data = pd.read_table(
                filepath,
                **InputData.new_format
            )
        else:
            raise RuntimeError("Unknown csv format {}".format(csv_format))

        csv_data.drop_duplicates(subset="label", keep=False, inplace=True)
        return csv_data

    @staticmethod
    def _merge_tubes(files: {str: str}, tubes) -> pd.DataFrame:
        '''Merge results from different tubes joining on label.
        Labels only contained in some of the tubes will be excluded in the join
        operation.'''
        # filter which tubes to read
        if not tubes:
            tubes = list(files.keys())

        # ensure that the tube list is properly sorted
        tubes = sorted(tubes)

        # read all tubes
        dataframes = [
            InputData._read_file(files[tube]) for tube in tubes
        ]

        non_number_cols = [
            [c for c in df.columns if not c.isdigit()]
            for df in dataframes
        ]
        not_both = reduce(lambda x, y: set(x) ^ set(y), non_number_cols)
        assert not not_both or len(dataframes) == 1, \
            "Different non-number columns in entered dataframes."
        metacols = non_number_cols[0]
        merged = reduce(
            lambda x, y: merge_on_label(x, y, metacols), dataframes
        )
        return merged

    def filter_data(self, *args, **kwargs):
        """Explicitly return a DataView object."""
        obj = super().filter_data(*args, **kwargs)
        return DataView.from_obj(obj)


class DataCollection:
    """Class wrapping multiple data sources, such as multiple SOM
    iterations.
    Individual data is saved in InputData objects.
    """

    def __init__(self, filesdict: dict, name: str, tubes: list):
        """Enter a mapping between names and dicts mapping tubes to files."""
        # load data into a dict mapping names with InputData objects
        self.name = name
        self._tubes = tubes
        self._filesdict = filesdict
        self._names = list(self._filesdict.keys())
        self._data = {}

        self._current_key = 0

    def __iter__(self):
        self._current_key = 0
        return self

    def __next__(self):
        key = self._current_key

        if key >= len(self._names):
            raise StopIteration

        self._current_key += 1
        return self[self._names[key]]

    def __getitem__(self, value):
        return InputData.from_files(
            self._filesdict[value], name=value, tubes=self._tubes
        )
