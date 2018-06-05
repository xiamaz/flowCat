'''
CSV file operations needed for upsampling based classification
'''
import re
import logging
from functools import reduce
from typing import Tuple

import pandas as pd

from lib.types import FilesDict, GroupSelection, SizeOption, MaybeList


RE_TUBE = re.compile(r"tube(\d+)\.csv")


def merge_on_label(left_df: pd.DataFrame, right_df: pd.DataFrame, metacols) \
        -> pd.DataFrame:
    '''Merge two dataframes on label column and drop additional replicated
    columns such as group.'''
    merged = left_df.merge(
        right_df, how="inner", on="label", suffixes=("", "_y")
    )
    merged.drop([
        s+"_y" for s in metacols if s != "label"
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


class DataView:
    '''Contains upsampling data with possiblity to apply filters, keeping
    state.
    '''

    def __init__(self, data: pd.DataFrame):
        self._data = data
        self._groups, self.group_names = self.split_groups(self._data)

    def get_test_train_split(self, ratio: float = None, abs_num: int = None) \
            -> (pd.DataFrame, pd.DataFrame):
        '''Test and training split for data evaluation.
        Ratio and abs_num are applied per cohort (eg unique label).
        '''
        if not (ratio or abs_num):
            raise TypeError("Neither ratio nor absolute number cutoff given")

        test_list = []
        train_list = []
        for group_name, group in self._groups:
            logging.info("Splitting %s", group_name)
            i = (group.shape[0] - abs_num) \
                if abs_num else int(ratio * group.shape[0])
            if i < 0:
                raise RuntimeError("Cutoff below zero. Perhaps small cohort.")
            shuffled = group.sample(frac=1)
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
                end = block_size * (i+1) if i+1 < k_num else group.shape[0]
                df_group.append(shuffled.iloc[start:end, :])
            df_list.append(df_group)
        df_splits = [pd.concat(dfs) for dfs in zip(*df_list)]
        df_splits = [df.sample(frac=1) for df in df_splits]
        return df_splits

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

    @staticmethod
    def split_groups(data):
        '''Split data into separate groups and binarizer functions and group
        name lists for easier processing.
        '''
        groups = list(data.groupby("group"))
        group_names = [n for n, _ in groups]
        return groups, group_names


class UpsamplingData:
    '''Data from upsampling in pandas dataframe form.'''

    def __init__(self, dataframe: pd.DataFrame):
        self._data = dataframe

    @classmethod
    def from_files(
            cls, files: FilesDict, tubes: MaybeList = None
    ) -> "UpsamplingData":
        '''Create upsampling data from structure containing multiple tubes
        for multiple files.
        The input structure contains a list of items to be joined by row,
        which are in turn joined on columns. (eg multiple tubes on the inner
        level split into multiple output files on the outer level)
        '''
        merged_tubes = [
            cls._merge_tubes([f[t] for t in tubes]) for f in files.values()
        ]
        metacols = [
            meta for _, meta in merged_tubes
        ]
        assert (reduce(lambda x, y: set(x) ^ set(y), metacols)), \
            "Different metacols in tubes."
        merged_dfs = [
            data for data, _ in merged_tubes
        ]
        return cls(pd.concat(merged_dfs))

    def get_data(self):
        '''Get unfiltered data as dataframe.'''
        return self._data

    def get_group_sizes(self):
        '''Get number of rows per group.'''
        return self._group_sizes(self._data)

    def get_group_names(self):
        """Get cohort names."""
        return self._data["group"].unique()

    def filter_data(
            self, groups: GroupSelection, cutoff: SizeOption, max_size: SizeOption
    ) -> DataView:
        '''Filter data based on criteria and return a data view.'''
        data = self._data
        if groups:
            data = self._select_groups(data, groups)
        if cutoff or max_size:
            if not isinstance(cutoff, dict):
                cutoff = {g: cutoff for g in self.get_group_names()}
            if not isinstance(max_size, dict):
                max_size = {g: max_size for g in self.get_group_names()}

            data = self._limit_size_per_group(
                data, lower=cutoff, upper=max_size
            )
        return DataView(data)

    @staticmethod
    def _select_groups(data: pd.DataFrame, groups: GroupSelection):
        '''Select only certain groups.'''
        for group in groups:
            if (
                    not len(group["tags"]) == 1
                    or not group["name"] == group["tags"][0]
            ):
                data.loc[
                    data["group"].isin(group["tags"]),
                    "group"
                ] = group["name"]
        group_names = [group["name"] for group in groups]
        data = data.loc[data["group"].isin(group_names)]
        return data

    @staticmethod
    def _limit_size_per_group(data: pd.DataFrame, lower: dict, upper: dict):
        '''Limit size per group.
        Lower limit of size - group will be excluded if smaller
        Upper limit of size - group will be downsampled if larger
        '''
        new_data = []
        for name, group in data.groupby("group"):
            if name in lower and group.shape[0] < lower[name]:
                continue
            if name in upper and group.shape[0] > upper[name]:
                group = group.sample(n=upper[name])
            new_data.append(group)
        return pd.concat(new_data)

    @staticmethod
    def _group_sizes(data: pd.DataFrame) -> pd.Series:
        return data.groupby("group").size()

    @staticmethod
    def _read_file(filepath: str) -> pd.DataFrame:
        '''Read output from preprocessing in csv format.
        '''
        csv_data = pd.read_table(filepath, sep=";", index_col=0)
        return csv_data

    @staticmethod
    def _merge_tubes(files: Tuple[str]) -> pd.DataFrame:
        '''Merge results from different tubes joining on label.
        Labels only contained in some of the tubes will be excluded in the join
        operation.'''
        dataframes = [
            UpsamplingData._read_file(fp) for fp in files
        ]
        non_number_cols = [
            [c for c in df.columns if not c.isdigit()]
            for df in dataframes
        ]
        not_both = reduce(lambda x, y: set(x) ^ set(y), non_number_cols)
        assert not not_both, \
            "Different non-number columns in entered dataframes."
        metacols = non_number_cols[0]
        merged = reduce(
            lambda x, y: merge_on_label(x, y, metacols), dataframes
        )
        return merged, metacols

    def __repr__(self) -> str:
        return repr(self._data)


def main():
    '''Tests based on using the joined upsampling tests.
    '''
    UpsamplingData.from_files([("../joined/cll_normal.csv")])


if __name__ == '__main__':
    main()
