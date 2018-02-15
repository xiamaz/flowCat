import os
import itertools
import functools
import random

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Conv1D
from keras.utils import to_categorical,plot_model
from keras import regularizers

import matplotlib
import seaborn
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
Matrix based classification
---
Classification using matrix information from FlowSOM data. Instead of histograms use individually generated cluster signatures inside the self-organizing-map.
Histogram is ordered by number of cells ordered to the specific cluster.
'''

class Groups():
    '''Class containing FileGroup correponding to different cohorts.
    '''
    def __init__(self, folder):
        self.name = os.path.split(folder)[1]
        self._groups = { f : FileGroup(os.path.join(folder,f)) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f)) }

    def keys(self):
        return self._groups.keys()

    def __getitem__(self, key):
        return self._groups[key]

    def __setitem__(self, key, value):
        self._groups[key] = value

    def __iter__(self):
        self.cur_index = 0
        return self

    def __next__(self):
        if self.cur_index >= len(self._groups):
            raise StopIteration
        r = self._groups[self.cur_index]
        self.cur_index += 1
        return r

    def __repr__(self):
        return repr(self._groups)

class SampleIndexer():
    def __init__(self, obj, *nsplit):
        self._obj = obj
        nn = list(range(0,len(self._obj)))
        random.shuffle(nn)
        self._samples = []
        i = 0
        for ns in nsplit:
            if ns >= len(nn):
                raise IndexError
            self._samples.append(nn[i:ns])
            i = ns
        self._samples.append(nn[i:])

    def __getitem__(self, key):
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
        r = self.__getitem__(self._i)
        self._i += 1
        return r

    def __repr__(self):
        return "Samples: {}".format(":".join([len(s) for s in self._samples]))


class FileGroup():
    '''Represents a collection of csv files. Data content is loaded on demand.
    '''
    def __init__(self, parentfolder, file_ext='.csv'):
        self._group = os.path.split(parentfolder)[1]
        self._parentfolder = parentfolder
        self._files = {os.path.splitext(f)[0]:f for f in os.listdir(parentfolder) if os.path.splitext(f)[1] == file_ext}
        self._cache = {}

    def sample_split(self, *nsplit):
        '''Split the files in such a way, that the first n groups will have the given number. The remaining will be put in the last class.
        '''
        nn = random.shuffle(list(range(0,len(self._files))))
        self.samples = SampleIndexer(self, *nsplit)
        return self

    def load_file(self, f):
        '''Reading files with on-demand caching.
        '''
        if f in self._cache:
            df = self._cache[f]
        else:
            filepath = os.path.join(self._parentfolder, f)
            df = pd.read_table(filepath, sep='\t', quotechar='"')
            self._cache[f] = df
        return df

    def __getitem__(self, key):
        '''Load dataframes of information matrices by index or otherwise load multiple selections in a list.
        '''
        if isinstance(key, str):
            return self.load_file(self._files[key])
        elif isinstance(key, int):
            return self.load_file(list(self._files.values())[key])
        elif isinstance(key, list) or isinstance(key, tuple):
            dfs = [ self.__getitem__(k) for k in key ]
            return dfs
        else:
            raise TypeError

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index >= len(self._files):
            raise StopIteration
        r = list(self._files.values())[self.iter_index]
        self.iter_index += 1
        df = pd.read_table(r)
        return df

    def __repr__(self):
        return "{} entries".format(len(self._files))

    def __len__(self):
        return len(self._files)

class MatrixClassifier:
    '''Neural network for classification of flowSOM cluster signatures.
    '''

    def __init__(self, groups):
        self.groups = groups

    def binary(self, positive, negative):
        if isinstance(positive, str):
            positive = [positive]
        if isinstance(negative, str):
            negative = [negative]
        self.positive = positive
        self.negative = negative

        neg_groups = [ self.groups[g] for g in negative ]
        pos_groups = [ self.groups[g] for g in positive ]

        # 80 - 20 split, hardcoded for now
        neg_groups = [ s.sample_split(int(len(s) * 0.8)) for s in neg_groups ]
        pos_groups = [ s.sample_split(int(len(s) * 0.8)) for s in pos_groups ]

        # extract information and flatten list
        neg_train = [ s.samples[0] for s in neg_groups ]
        neg_train = [ t for s in neg_train for t in s ]
        neg_test = [ s.samples[1] for s in neg_groups ]
        neg_test = [ t for s in neg_test for t in s ]
        pos_train = [ s.samples[0] for s in pos_groups ]
        pos_train = [ t for s in pos_train for t in s ]
        pos_test = [ s.samples[1] for s in pos_groups ]
        pos_test = [ t for s in pos_test for t in s ]

        model = Sequential()
        model.add(Conv1D(1, 3, input_shape=(100,13)))

    def multiclass(self):
        pass

def main():
    data_folder = '/home/max/DREAM/flowCat/flowdensity/csvs'

    all_groups = Groups(data_folder)

    classif = MatrixClassifier(all_groups)

    classif.binary('FL', 'DLBCL')




if __name__ == '__main__':
    main()
