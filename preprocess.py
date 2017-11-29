#!/usr/bin/python3
import fcsparser
import pandas
import numpy
import os
import re
from pprint import pprint
import collections
import itertools
from argparse import ArgumentParser

from sklearn.decomposition import PCA

# preprocessing in python by simply generating some distribution identifiers

ID_RE = re.compile('^([KMPB\d-]+)(.*)LMD$')
ID_CELL = re.compile('^([KMPB\d-]+) CLL 9F (\d+).*.LMD$')

BLACKLIST = [
        'DLBCL'
        ]

def file_structure(path):
    '''
    Acquire dataframe with fcs files. Use directory structure as ordering.

    Args:
        path: Path of top directory. Child directories will be interpreted as classes.

    Returns:
        Dataframe of filenames with additional information.
    '''
    dict_array = {
            'group' : []
            ,'id' : []
            ,'set' : []
            ,'filename' : []
            }
    for f in os.listdir(path):
        for i in os.listdir(os.path.join(path, f)):
            if os.path.getsize(os.path.join(path, f, i)) == 0:
                open('error_files.txt', 'a').write('Size 0 file {}\n'.format(i))
                continue
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
        try:
            flow_frames.append(fcsparser.parse(f, data_set=0))
        except ValueError:
            print("Corrupted file: {}".format(f))
            continue
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

def load(path, setnum=1, negative='normal control', positive='CLL'):
    files = file_structure(path)
    print(files)


def count_id(id_col):
    uniq = id_col.unique()
    return len(uniq)

def confirm_experiment_set(fileset):
    '''
    Tests for the given fileset, whether they contain the same set of fcs experiments.
    Different ids should have an identical set of measuring samples.

    Args:
        fileset: dataframe containing at least group, id and flowframe

    Returns:
        Boolean, whether set is same or not
    '''
    print("Tube headers for every sample")
    sample_bins = collections.defaultdict(list)
    sigs = []
    for i, f in fileset.iterrows():
        col_sig = f['flowframe'][1].columns.values.tolist()
        if col_sig not in sigs:
            sigs.append(col_sig)
        col_str = "|".join(col_sig)
        sample_bins[col_str].append(f['id'])
    consensus_labels = None
    matches = True

    # check for interchangeable
    for s in sigs:
        for l in sigs:
            # skip if superficially similar
            if "".join(s) == "".join(l):
                continue
            misses = 0
            for w in s:
                if w not in l:
                    misses += 1
            if misses == 0:
                print('{}\nis contained in\n{}'.format(s, l))
                sample_bins["|".join(s)] += sample_bins.pop("|".join(l))

    # we can merge bins by removing the additional columns in the larger one later
    print("\n".join(sample_bins.keys()))
    for i in fileset['id']:
        for s,v in sample_bins.items():
            if i not in v:
                print("{} does not have\n{}".format(i,s))
                matches = False
    if matches:
        print("All samples contain the same set of experiments.")
    return matches

def special_stats(file_structure):
    subset = file_structure[file_structure['group'] != BLACKLIST[0]]
    subset_fl = read_fcs(subset)
    confirm_experiment_set(subset_fl)

def data_statistics(file_structure):
    '''
    Returns information about number of files per group, number of unique ids, files per id.

    Args:
        file_structure: Dataframe structure with at least filename, group and id

    Returns:
        Dataframe with statistic information per group
    '''
    groups = file_structure.groupby('group')
    num = groups['id'].apply(count_id)
    print("Number of unique ids per group")
    print(num)
    ## work on a smaller subset first for speed reasons
    tube12 = file_structure.loc[file_structure['set'].isin([1,2])]
    groups = tube12['group'].unique()
    groups = [ g for g in groups if g not in BLACKLIST ]

    for group in groups:
        print("{} -----------------------".format(group))
        subset = file_structure[file_structure['group'] == group]
        subset_fl = read_fcs(subset)
        confirm_experiment_set(subset_fl)

## using principal component analysis for dimensionality reduction
def main():
    parser = ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    files = file_structure(args.directory)
    data_statistics(files)

if __name__ == '__main__':
    main()
