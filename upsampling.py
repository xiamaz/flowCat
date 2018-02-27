'''
CSV file operations needed for upsampling based classification
'''
import logging
from typing import Union, Callable
import os
import random

import pandas as pd
import numpy as np


def create_binarizer(names):
    '''Create and parse binary label descriptions
    '''
    def binarizer(factor):
        value = names.index(factor)
        # bin_array = np.zeros(len(names))
        bin_list = [0] * len(names)
        # bin_array[value] = 1
        bin_list[value] = 1
        return pd.Series(bin_list)

    def debinarizer(bin_array):
        val = np.argmax(bin_array)
        return names[val]

    return binarizer, debinarizer


class UpsamplingData:
    '''Data from upsampling in pandas dataframe form.'''

    def __init__(self, dataframe: pd.DataFrame):
        self._load_data(dataframe)

    def add_data_from_file(self, filepath: str):
        new_data = self._read_file(filepath)
        self._load_data(new_data)

    def get_test_train_split(self, ratio: float=None, abs_num: int=None) \
            -> (pd.DataFrame, pd.DataFrame):
        if not (ratio or abs_num):
            raise TypeError("Neither ratio nor absolute number cutoff given")

        test_list = []
        train_list = []
        for group_name, group in self._groups:
            logging.info("Splitting", group_name)
            i = ratio and int(ratio * group.shape[0]) or abs_num
            shuffled = group.sample(frac=1)
            test, train = shuffled.iloc[:i, :], shuffled.iloc[i:, :]
            test_list.append(test)
            train_list.append(train)

        df_test = pd.concat(test_list)
        df_test = df_test.sample(frac=1)
        df_train = pd.concat(train_list)
        df_train = df_train.sample(frac=1)
        return df_test, df_train

    def _load_data(self, data: pd.DataFrame):
        if hasattr(self, "data"):
            self._data = pd.concat([self._data, data])
        else:
            self._data = data
        self._split_groups()

    def _split_groups(self):
        self._groups = self._data.groupby("group")
        # create closures for label creation
        self.binarizer, self.debinarizer = self.binarizers(self._data['group'])

    @classmethod
    def from_file(cls, filepath: str) -> "UpsamplingData":
        return cls(cls._read_file(filepath))

    @staticmethod
    def _read_file(filepath: str) -> pd.DataFrame:
        csv_data = pd.read_table(filepath, sep=";")
        return csv_data

    @staticmethod
    def binarizers(col_data: pd.Series) -> (Callable, Callable):
        '''Create binarizer and debinarizer from factors in a pandas series.
        '''
        return create_binarizer(list(col_data.unique()))

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
