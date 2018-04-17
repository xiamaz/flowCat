'''
CSV file operations needed for upsampling based classification
'''
import logging
from functools import reduce
from typing import Callable, Tuple

import pandas as pd
import numpy as np

from lib.types import FilesDict, GroupSelection


def merge_on_label(left_df: pd.DataFrame, right_df: pd.DataFrame) \
        -> pd.DataFrame:
    '''Merge two dataframes on label column and drop additional replicated
    columns such as group.'''
    merged = left_df.merge(right_df, how="inner", on="label", suffixes=("",
                                                                        "_y"))
    merged.drop(["group_y"], inplace=True, axis=1)
    # continuous field naming
    names = [m for m in list(merged) if m != "group" and m != "label"]
    rename_dict = {m: str(i + 1) for i, m in enumerate(names)}
    merged.rename(columns=rename_dict, inplace=True)
    # reorder the columns
    merged_names = list(rename_dict.values()) + ["group", "label"]
    merged = merged[merged_names]
    return merged


def create_binarizer(names):
    '''Create and parse binary label descriptions
    '''
    def binarizer(factor):
        '''Create binary array from labels.
        '''
        value = names.index(factor)
        # bin_array = np.zeros(len(names))
        bin_list = [0] * len(names)
        # bin_array[value] = 1
        bin_list[value] = 1
        return pd.Series(bin_list)

    def debinarizer(bin_array, matrix=True):
        '''Get label from binary array or position number.
        '''
        if matrix:
            val = np.argmax(bin_array)
        else:
            val = bin_array
        return names[val]

    return binarizer, debinarizer


class UpsamplingData:
    '''Data from upsampling in pandas dataframe form.'''

    def __init__(self, dataframe: pd.DataFrame):
        self._data = dataframe
        self._groups, (self.binarizer, self.debinarizer), self.group_names = \
            self.split_groups(self._data)

    def update_data(self, data: pd.DataFrame):
        '''Update internal data with new data and refresh the group
        representations.
        '''
        self._data = data
        self._groups, (self.binarizer, self.debinarizer), self.group_names = \
            self.split_groups(self._data)

    def select_groups(self, groups: GroupSelection):
        '''Select only certain groups.'''
        data = self._data

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

        # limit data to ones inside group
        self.update_data(data.loc[data['group'].isin(group_names)])

    def exclude_small_cohorts(self, cutoff=50):
        '''Exclude cohorts below threshold and report these.'''
        selected_groups = [
            group for gn, group in self._groups
            if group.shape[0] >= cutoff
        ]
        excluded_groups = {
            gn: group.shape[0] for gn, group in self._groups
            if group.shape[0] < cutoff
        }
        print("Excluded groups: ", excluded_groups)
        self.update_data(pd.concat(selected_groups))

    def limit_size(self, size):
        '''Limit size of cohorts to specified size. If cohorts are smaller
        than this size, they will not be changed.'''
        sample_data = [
            group.sample(n=min(group.shape[0], size))
            for group_name, group in self._groups
        ]
        self.update_data(pd.concat(sample_data))

    def limit_size_to_smallest(self):
        '''Limit size of all cohorts to smallest.'''
        min_size = min([group.shape[0] for gn, group in self._groups])
        self.limit_size(min_size)

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

    @classmethod
    def from_files(cls, files: FilesDict) -> "UpsamplingData":
        '''Create upsampling data from structure containing multiple tubes
        for multiple files.
        The input structure contains a list of items to be joined by row,
        which are in turn joined on columns. (eg multiple tubes on the inner
        level split into multiple output files on the outer level)
        '''
        merged_tubes = [cls.merge_tubes(f) for f in files.values()]
        return cls(pd.concat(merged_tubes))

    @staticmethod
    def read_file(filepath: str) -> pd.DataFrame:
        '''Read output from preprocessing in csv format.
        '''
        csv_data = pd.read_table(filepath, sep=";")
        return csv_data

    @staticmethod
    def merge_tubes(files: Tuple[str]) -> pd.DataFrame:
        '''Merge results from different tubes joining on label.
        Labels only contained in some of the tubes will be excluded in the join
        operation.'''
        dataframes = [UpsamplingData.read_file(fp) for fp in files]
        merged = reduce(merge_on_label, dataframes)
        return merged

    @staticmethod
    def binarizers(col_data: pd.Series) -> (Callable, Callable):
        '''Create binarizer and debinarizer from factors in a pandas series.
        '''
        group_names = sorted(list(col_data.unique()))
        binarizer, debinarizer = create_binarizer(group_names)
        return binarizer, debinarizer, group_names

    @staticmethod
    def split_x_y(dataframe: pd.DataFrame, binarizer: Callable) \
            -> (np.matrix, np.matrix):
        '''Split dataframe into matrices with group labels as sparse matrix'''
        x_matrix = dataframe.drop(['group', 'label'], axis=1).as_matrix()
        y_matrix = dataframe['group'].apply(binarizer).as_matrix()
        return x_matrix, y_matrix

    @staticmethod
    def split_groups(data):
        '''Split data into separate groups and binarizer functions and group
        name lists for easier processing.
        '''
        groups = list(data.groupby("group"))
        # create closures for label creation
        binarizer, debinarizer, group_names = UpsamplingData.binarizers(
            data['group'])
        return groups, (binarizer, debinarizer), group_names

    def __repr__(self) -> str:
        return repr(self._data)


def main():
    '''Tests based on using the joined upsampling tests.
    '''
    data = UpsamplingData.from_files([("../joined/cll_normal.csv")])
    test, _ = data.get_test_train_split(abs_num=60)
    data.split_x_y(test, data.binarizer)


if __name__ == '__main__':
    main()
