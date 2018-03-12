'''
CSV file operations needed for upsampling based classification
'''
import logging
from functools import reduce
from typing import Callable

import pandas as pd
import numpy as np


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
        self._load_data(dataframe)

    def add_data_from_file(self, filepath: str):
        '''Add additional rows to existing dataframe. Should have same column
        names.'''
        new_data = self._read_file(filepath)
        self._load_data(new_data)

    def select_groups(self, groups):
        '''Select only certain groups.'''
        self._data = self._data.loc[self._data['group'].isin(groups)]
        self._split_groups()

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
        self._data = pd.concat(selected_groups)
        self._split_groups()

    def limit_size_to_smallest(self):
        '''Limit size of all cohorts to smallest.'''
        min_size = min([group.shape[0] for gn, group in self._groups])
        print(min_size)
        sample_data = [
            group.sample(n=min_size)
            for group_name, group in self._groups
        ]
        self._data = pd.concat(sample_data)
        self._split_groups()

    def get_test_train_split(self, ratio: float=None, abs_num: int=None) \
            -> (pd.DataFrame, pd.DataFrame):
        if not (ratio or abs_num):
            raise TypeError("Neither ratio nor absolute number cutoff given")

        test_list = []
        train_list = []
        for group_name, group in self._groups:
            logging.info("Splitting", group_name)
            abs_cutoff = abs_num and group.shape[0] - abs_num or abs_num
            i = ratio and int(ratio * group.shape[0]) or abs_cutoff
            if (i < 0):
                raise RuntimeError("Cutoff below zero. Perhaps small cohort.")
            shuffled = group.sample(frac=1)
            test, train = shuffled.iloc[:i, :], shuffled.iloc[i:, :]
            test_list.append(test)
            train_list.append(train)

        df_test = pd.concat(test_list)
        df_test = df_test.sample(frac=1)
        df_train = pd.concat(train_list)
        df_train = df_train.sample(frac=1)
        return df_test, df_train

    def k_fold_split(self, k_num: int=5) -> [pd.DataFrame]:
        '''Split file into k same-size partitions. Keep the group ratios
        same.
        '''
        df_list = []
        for group_name, group in self._groups:
            shuffled = group.sample(frac=1)
            df_group = []
            for i in range(k_num):
                block_size = int(group.shape[0] / k_num)
                start = block_size * i
                end = min(block_size * (i+1), group.shape[0])
                df_group.append(shuffled.iloc[start:end, :])
            df_list.append(df_group)
        df_splits = [pd.concat(dfs) for dfs in zip(*df_list)]
        df_splits = [df.sample(frac=1) for df in df_splits]
        return df_splits

    def _load_data(self, data: pd.DataFrame):
        if hasattr(self, "data"):
            self._data = pd.concat([self._data, data])
        else:
            self._data = data
        self._split_groups()

    def _split_groups(self):
        self._groups = list(self._data.groupby("group"))
        # create closures for label creation
        self.binarizer, self.debinarizer, self.group_names = self.binarizers(
            self._data['group'])

    @classmethod
    def from_file(cls, filepath: str) -> "UpsamplingData":
        return cls(cls._read_file(filepath))

    @classmethod
    def from_multiple_tubes(cls, filepaths: [str]) -> "UpsamplingData":
        '''Create information from multiple tubes by joining dataframes.'''
        dataframes = [cls._read_file(fp) for fp in filepaths]
        merged = reduce(merge_on_label, dataframes)
        return cls(merged)

    @staticmethod
    def _read_file(filepath: str) -> pd.DataFrame:
        csv_data = pd.read_table(filepath, sep=";")
        return csv_data

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

    def __repr__(self) -> str:
        return repr(self._data)


def main():
    '''Tests based on using the joined upsampling tests.
    '''
    data = UpsamplingData.from_file("../joined/cll_normal.csv")
    test, train = data.get_test_train_split(abs_num=60)
    x_matrix, y_matrix = data.split_x_y(test, data.binarizer)


if __name__ == '__main__':
    main()
