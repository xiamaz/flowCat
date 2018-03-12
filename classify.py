'''
Neural network classification using keras
'''
# from typing import Callable
import os
from typing import List, Tuple

from lib.upsampling import UpsamplingData
from lib.classification import Classifier
# from lib import plotting


def evaluate(files: List[Tuple[str]], name: str) -> None:
    '''Evaluate upsampling data.'''
    data = UpsamplingData.from_files(files)
    data.select_groups(["CLL", "normal", "CLLPL", "HZL",
                        "Mantel", "Marginal", "MBL"])
    # data.limit_size_to_smallest()
    clas = Classifier(data, name=name)
    clas.holdout_validation(ratio=0.8)
    # clas.k_fold_validation(k_num=5)


def main():
    '''Classification of single tubes
    '''
    files = [("output/preprocess/native_tube1.csv",
              "output/preprocess/native_tube2.csv")]
    evaluate(files, "network_analysis")


if __name__ == '__main__':
    main()
