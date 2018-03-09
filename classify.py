'''
Neural network classification using keras
'''
# from typing import Callable

from lib.upsampling import UpsamplingData
from lib.classification import Classifier
# from lib import plotting


def joined_tubes(files: [str], name: str):
    '''Classification based on tubes joined on SOM level.'''
    data = UpsamplingData.from_file(files[0])
    data.select_groups(["CLL", "normal", "CLLPL", "HZL",
                        "Mantel", "Marginal", "MBL"])
    clas = Classifier(data)
    # clas.holdout_validation(name="joined", ratio=0.9)
    # clas.absolute_validation(name="joined")
    clas.k_fold_validation(name=name, k_num=5)


def separate_tubes(files: [str], name: str):
    '''Classification of separately upsampled tubes, which need to be joined
    for neural network classification.'''
    data = UpsamplingData.from_multiple_tubes(files)
    data.select_groups(["CLL", "normal", "CLLPL", "HZL",
                        "Mantel", "Marginal", "MBL"])
    clas = Classifier(data)

    # clas.holdout_validation(name="merged_all", ratio=0.9)
    # clas.absolute_validation(name="plain")
    clas.k_fold_validation(name=name, k_num=5)


def main():
    '''Classification of single tubes
    '''
    files_plain = [
        "/data/ssdraid/Genetik/PlainNormal_1_output/native_tube1.csv",
        "/data/ssdraid/Genetik/PlainNormal_1_output/native_tube2.csv"
    ]
    files_margin_all = [
        "/data/ssdraid/Genetik/MarginUpperLower_1_output/native_tube1.csv",
        "/data/ssdraid/Genetik/MarginUpperLower_1_output/native_tube2.csv"
    ]
    files_margin_upper = [
        "/data/ssdraid/Genetik/MarginUpper_1_output/native_tube1.csv",
        "/data/ssdraid/Genetik/MarginUpper_1_output/native_tube2.csv",
    ]
    files_margin_lower = [
        "/data/ssdraid/Genetik/MarginLower_1_output/native_tube1.csv",
        "/data/ssdraid/Genetik/MarginLower_1_output/native_tube2.csv",
    ]
    files_joined = [
        "/data/ssdraid/Genetik/joined/joined_all.csv"
    ]
    # separate_tubes(files_plain, "plain")
    # separate_tubes(files_margin_all, "margin_all")
    # separate_tubes(files_margin_upper, "margin_upper")
    # separate_tubes(files_margin_lower, "margin_lower")
    joined_tubes(files_joined, "joined")


if __name__ == '__main__':
    main()
