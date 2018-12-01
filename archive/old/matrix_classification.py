"""
Matrix based classification
---
Classification using matrix information from FlowSOM data. Instead of
histograms use individually generated cluster signatures inside the
self-organizing-map.  Histogram is ordered by number of cells ordered to the
specific cluster.
"""

import os
import random
from typing import Union

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten


def split_list(data: list, ratio: float):
    '''Split a list into two with the defined ratio
    '''
    split_point = int(len(data) * ratio)
    data1, data2 = data[:split_point], data[split_point:]
    return data1, data2


def create_test_train(data: {str: [tuple]}, ratio: float):
    '''Create train and test set from data.
    '''
    # shuffle data and split
    [random.shuffle for v in data.values()]
    data = {k: split_list(v, ratio) for k, v in data.items()}
    # put our grouped data into two continuous lists
    train, test = zip(*data.values())
    test = [y for x in test for y in x]
    train = [y for x in train for y in x]
    random.shuffle(test)
    random.shuffle(train)
    return train, test


class SampleIndexer():
    '''Implements indexing for the FileGroup object.
    This class enables splitting into test, training and validation sets.
    '''
    def __init__(self, obj: 'FileGroup', *nsplit: [int]):
        self._obj = obj
        # get a list of indices in the parent object, which we can shuffle
        index_list = list(range(0, len(self._obj)))
        random.shuffle(index_list)
        self._samples = []
        i = 0
        for split_index in nsplit:
            if split_index >= len(index_list):
                raise IndexError
            self._samples.append(index_list[i:split_index])
            i = split_index
        self._samples.append(index_list[i:])

    def __getitem__(self, key: str):
        if isinstance(key, int):
            return self._obj[self._samples[key]]
        else:
            raise TypeError

    def __iter__(self):
        '''Iterating over samples will return entire sets at a time.
        '''
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self._samples):
            raise StopIteration
        item = self.__getitem__(self._i)
        self._i += 1
        return item

    def __repr__(self):
        return "Samples: {}".format(":".join([len(s) for s in self._samples]))


class FileGroup():
    '''Represents a collection of csv files. Data content is loaded on demand.
    '''
    def __init__(self, parentfolder: str, file_ext='.csv'):
        self._group = os.path.split(parentfolder)[1]
        self._parentfolder = parentfolder
        self._files = {os.path.splitext(f)[0]: f
                       for f in os.listdir(parentfolder)
                       if os.path.splitext(f)[1] == file_ext}
        self._cache = {}

    def sample_split(self, *nsplit: [int]):
        '''Split the files in such a way, that the first n groups will have
        the given number. The remaining will be put in the last class.
        '''
        self.samples = SampleIndexer(self, *nsplit)
        return self

    def _load_file(self, filename: str):
        '''Reading files with on-demand caching.
        '''
        if filename in self._cache:
            data = self._cache[filename]
        else:
            filepath = os.path.join(self._parentfolder, filename)
            data = pd.read_table(filepath, sep='\t', quotechar='"')
            self._cache[filename] = data
        return data

    def __getitem__(self, key: Union[str, slice, int, list, tuple]):
        '''Load dataframes of information matrices by index or otherwise load
        multiple selections in a list.
        '''
        if isinstance(key, str):
            return self.load_file(self._files[key])
        elif isinstance(key, slice):
            print(key)
            return self.__getitem__(list(range(key.start, key.stop, key.step)))
        elif isinstance(key, int):
            return self.load_file(list(self._files.values())[key])
        elif isinstance(key, list) or isinstance(key, tuple):
            return [self.__getitem__(k) for k in key]
        else:
            raise TypeError

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index >= len(self._files):
            raise StopIteration
        filename = list(self._files.values())[self.iter_index]
        self.iter_index += 1
        data = self._load_file(filename)
        return data

    def __repr__(self):
        return "{} entries".format(len(self._files))

    def __len__(self):
        return len(self._files)


class Groups():
    '''Class containing FileGroup correponding to different cohorts.
    '''
    def __init__(self, folder: str):
        self.name = os.path.split(folder)[1]
        self._groups = {f: FileGroup(os.path.join(folder, f)) for f in
                        os.listdir(folder) if
                        os.path.isdir(os.path.join(folder, f))}

    def keys(self):
        return self._groups.keys()

    def __getitem__(self, key: str):
        return self._groups[key]

    def __setitem__(self, key: str, value: FileGroup):
        self._groups[key] = value

    def __iter__(self):
        self.cur_index = 0
        return self

    def __next__(self):
        if self.cur_index >= len(self._groups):
            raise StopIteration
        response = self._groups[self.cur_index]
        self.cur_index += 1
        return response

    def __repr__(self):
        return repr(self._groups)


class MatrixClassifier:
    '''Neural network for classification of flowSOM cluster signatures.
    '''

    def __init__(self, groups: Groups):
        self.groups = groups

    def binary(self, positive: [str], negative: [str]):
        self.positive = positive
        self.negative = negative

        neg_groups = [self.groups[g] for g in negative]
        pos_groups = [self.groups[g] for g in positive]

        # flatten the list of groups of files
        neg_files = [f.as_matrix() for g in neg_groups for f in g]
        neg_result_matrix = np.zeros((len(neg_files), 2))
        neg_result_matrix[:, 0] = 1
        pos_files = [f.as_matrix() for g in pos_groups for f in g]
        pos_result_matrix = np.zeros((len(pos_files), 2))
        pos_result_matrix[:, 1] = 1
        # negative files with labeling as list
        all_files = {
            'negative': list(zip(neg_files, neg_result_matrix.tolist())),
            'positive': list(zip(pos_files, pos_result_matrix.tolist()))
        }
        train, test = create_test_train(all_files, 0.9)

        train_x, train_y = list(zip(*train))
        test_x, test_y = list(zip(*train))

        # 80 - 20 split, hardcoded for now
        # neg_groups = [s.sample_split(int(len(s) * 0.8)) for s in neg_groups]
        # pos_groups = [s.sample_split(int(len(s) * 0.8)) for s in pos_groups]

        # extract information and flatten list
        # neg_train = [s.samples[0] for s in neg_groups]
        # neg_train = [t for s in neg_train for t in s]
        # neg_train_labels = create_labels(len(neg_train), 2, 0)
        # neg_test = [s.samples[1] for s in neg_groups]
        # neg_test = [t for s in neg_test for t in s]
        # neg_test_labels = create_labels(len(neg_test), 2, 0)

        # pos_train = [s.samples[0] for s in pos_groups]
        # pos_train = [t for s in pos_train for t in s]
        # pos_train_labels = create_labels(len(pos_train), 2, 1)
        # pos_test = [s.samples[1] for s in pos_groups]
        # pos_test = [t for s in pos_test for t in s]
        # pos_test_labels = create_labels(len(pos_test), 2, 1)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        model = Sequential()
        model.add(Flatten(input_shape=(100, 13)))
        # model.add(Dense(1000, activation='relu'))
        # model.add(Dense(100, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adadelta',
                      metrics=['acc'])
        model.fit(x=train_x, y=train_y, batch_size=16, epochs=200)
        loss = model.evaluate(x=test_x, y=test_y, batch_size=64)
        print(loss)

    def multiclass(self):
        pass


def main():
    '''Main execution flow.
    '''

    data_folder = '/data/ssdraid/Genetik/flowCat/csvs'

    all_groups = Groups(data_folder)

    classif = MatrixClassifier(all_groups)

    classif.binary(['FL'], ['DLBCL'])


if __name__ == '__main__':
    main()
