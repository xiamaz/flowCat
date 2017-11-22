#!/usr/bin/python3
import fcsparser
import pandas
import numpy
import os
import re
from pprint import pprint
import collections
import itertools

from sklearn.decomposition import PCA

# preprocessing in python by simply generating some distribution identifiers

ID_RE = re.compile('^([KMPB\d-]+)(.*)LMD$')
ID_CELL = re.compile('^([KMPB\d-]+) CLL 9F (\d+).*.LMD$')

def file_structure(path):
    dict_array = {
            'group' : []
            ,'id' : []
            ,'set' : []
            ,'filename' : []
            }
    for f in os.listdir(path):
        for i in os.listdir(os.path.join(path, f)):
            m = ID_CELL.match(i)
            if m is None:
                continue
            dict_array['group'].append(f)
            dict_array['id'].append(m.group(1))
            dict_array['set'].append(int(m.group(2)))
            dict_array['filename'].append(os.path.join(path,f,i))
    for d in dict_array:
        dict_array[d] = numpy.array(dict_array[d])
    df = pandas.DataFrame.from_dict(dict_array)
    return df

def read_fcs(file_frame):
    flow_frames = []
    for f in file_frame['filename']:
        flow_frames.append(fcsparser.parse(f, data_set=0))
    flow_frames = pandas.Series(flow_frames)
    file_frame = file_frame.assign(flowframe=flow_frames.values)
    return file_frame

def scale_flowframes(flowframes):
    scaled_df = []
    for f in flowframes['flowframe']:
        # scaling for poor people
        meta, data = f
        data -= data.min()
        data /= data.max()
        scaled_df.append(data)
    return flowframes.assign(flowframe=pandas.Series(scaled_df).values)

## steps to take
# calculate some simple properties per datapoint

def stats(scaled):
    stats_dict = collections.defaultdict(list)
    for i,r in scaled.iterrows():
        stats_dict['id'].append(r['id'])
        stats_dict['label'].append(r['group'])
        fc = r['flowframe']
        mean = fc.mean()
        stats_dict['mean'].append(mean)
        std = fc.std()
        stats_dict['std'].append(std)
        skew = fc.skew()
        stats_dict['skew'].append(skew)
        kurtosis = fc.kurtosis()
        stats_dict['kurtosis'].append(kurtosis)
        median = fc.median()
        stats_dict['median'].append(median)
        iqr = fc.quantile (0.75) - fc.quantile(0.25)
        stats_dict['iqr'].append(iqr)
        # print("{} -- ø{} σ{} δ{} κ{} μ{} χ{}".format(n, mean, std, skew, kurtosis, median, iqr))
    method_names = []
    dfs = []
    for k in stats_dict:
        if k not in ['id', 'label']:
            stats_dict[k] = pandas.concat(stats_dict[k], axis=1)
            # ind = stats_dict[k].index
            method_names.append(k)
            dfs.append(stats_dict[k])
            # multi_names += list(zip(itertools.cycle([k]), ind))
    # mult = pandas.MultiIndex_from_tuples(multi_names, names=['method', 'marker'])
    stat_df = pandas.concat(dfs, keys=method_names).transpose()
    stat_df = stat_df.assign(label=stats_dict['label'])
    stat_df = stat_df.assign(id=stats_dict['id'])
    return stat_df

## using principal component analysis for dimensionality reduction
def main():
    path = '/home/max/DREAM/Krawitz'
    files = file_structure(path)
    set1 = read_fcs(files[files['set'] == 1])
    set1_normal = set1[set1['group'] == 'normal control']
    set1_cll = set1[set1['group'] == 'CLL']
    chosen_set = set1_normal
    scaled = scale_flowframes(chosen_set)
    stat_df = stats(scale_flowframes(chosen_set))


if __name__ == '__main__':
    main()
