'''
Neural network classification using keras
'''
# from typing import Callable
import os
from typing import List, Tuple

from lib.upsampling import UpsamplingData
from lib.classification import Classifier
# from lib import plotting


def evaluate(files: List[Tuple[str]], name: str, input_path: str):
    '''Evaluate upsampling data.'''
    for tube_files in files:


def joined_tubes(files: [str], name: str):
    '''Classification based on tubes joined on SOM level.'''
    data = UpsamplingData.from_file(files[0])
    # data.select_groups(["CLL", "normal", "CLLPL", "HZL",
    #                     "Mantel", "Marginal", "MBL"])
    clas = Classifier(data, name=name)
    # clas.holdout_validation(name="joined", ratio=0.9)
    # clas.absolute_validation(name="joined")
    clas.k_fold_validation(k_num=5)


def separate_tubes(files: [str], name: str):
    '''Classification of separately upsampled tubes, which need to be joined
    for neural network classification.'''
    data = UpsamplingData.from_multiple_tubes(files)
    data.select_groups(["CLL", "normal", "CLLPL", "HZL",
                        "Mantel", "Marginal", "MBL"])
    # data.exclude_small_cohorts(cutoff=50)
    data.limit_size_to_smallest()
    clas = Classifier(data, name=name)

    # clas.holdout_validation(name="merged_all", ratio=0.9)
    # clas.absolute_validation(name="plain")
    clas.k_fold_validation(k_num=5)


def main():
    '''Classification of single tubes
    '''
    input_folder = "output/preprocess"
    files = [("native_tube1.csv", "native_tube2.csv")]
    evaluate(files, "network_analysis", input_folder)


if __name__ == '__main__':
    main()
