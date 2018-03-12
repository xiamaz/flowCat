'''
Neural network classification using keras
'''
import os
import logging
from typing import Callable, List, Union, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import sklearn.metrics as skm
from keras.models import Sequential
from keras.layers import Dense

from lib.upsampling import UpsamplingData
from lib import plotting


def avg_dicts(dicts: [dict]) -> dict:
    '''Average a list of dicts.'''
    avg_dict = defaultdict(float)
    for item in dicts:
        for key, value in item:
            avg_dict[key] += value
    avg_dict = {k: v/len(dicts) for k, v in avg_dict.items()}
    return avg_dict


class Classifier:
    '''Basic sequential dense classifier.'''

    def __init__(self,
                 data: UpsamplingData,
                 output_path: str = "output/classification",
                 name: str = "expname"):
        self.data = data
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.experiment_name = name

    def k_fold_validation(self,
                          k_num: int = 5,
                          save_individual_results: bool = True):
        '''Do k-fold validation on data set.
        '''
        # put all output into a separate folder
        name_tag = "_kfold"

        splits = self.data.k_fold_split(k_num)
        eval_results = []
        for i in range(k_num):
            test = splits[i]
            train = pd.concat(splits[i+1:] + splits[:i])
            model = self.create_sequential_model(train, self.data.binarizer)
            result = self.evaluate_model(
                model, test, self.data.binarizer, self.data.debinarizer)
            eval_results.append(result)
            if save_individual_results:
                self.generate_output(*result, name_tag=name_tag)

        avg_confusion = sum([t[0] for t in eval_results])
        avg_stats = avg_dicts([t[1] for t in eval_results])
        self.generate_output(
            (avg_confusion, self.data.group_names),
            avg_stats,
            mismatches=None,
            name_tag=name_tag
        )

    def holdout_validation(self, ratio: float = 0.8,
                           abs_num: int = None):
        '''Simple holdout validation. If given abs_num, then each test cohort
        will contain the absolute number of cases.'''
        if abs_num:
            name_tag = "_absolute"
            train, test = self.data.get_test_train_split(abs_num=abs_num)
        else:
            name_tag = "_holdout"
            train, test = self.data.get_test_train_split(ratio=ratio)

        model = self.create_sequential_model(
            train, self.data.binarizer)

        confusion, stats, mismatches = self.evaluate_model(
            model, test, self.data.binarizer, self.data.debinarizer)

        self.generate_output((confusion, self.data.group_names),
                             stats, mismatches, name_tag)

    def generate_output(
            self,
            confusion_data: Union[Tuple[np.matrix, List[str]], None],
            statistics: Union[dict, None],
            mismatches: Union[dict, None],
            name_tag: str) -> None:
        '''Create confusion matrix and text statistics and save them to the
        output folder.
        '''
        tagged_name = self.experiment_name + name_tag

        if confusion_data:
            logging.info("Create confusion matrix plot.")
            plot_name = os.path.join(self.output_path,
                                     tagged_name + "_confusion.png")
            plotting.plot_confusion_matrix(*confusion_data,
                                           normalize=False,
                                           filename=plot_name)
        if statistics:
            logging.info("Saving statistics information to text file.")
            stat_file = os.path.join(self.output_path,
                                     self.experiment_name + "_statistics.txt")
            open(stat_file, "w").write(
                "{}\n====\n{}".format(tagged_name, statistics))

        if mismatches:
            logging.info("Creating text list from mismatch information.")
            mism_file = os.path.join(self.output_path,
                                     tagged_name + "_mismatches.txt")
            mismatch_lines = [
                "{}: T<{}>|F<{}>".format(
                    label, value["true"], value["predicted"])
                for label, value in mismatches
            ]
            with open(mism_file, "w") as mismatch_output:
                mismatch_output.writelines(mismatch_lines)

    @staticmethod
    def create_sequential_model(training_data: "DataFrame",
                                binarizer: Callable) -> Sequential:
        '''Create a sequential neural network with specified hidden layers.
        The input and output dimensions are inferred from the given
        data and labels. (Labels are converted to binary matrix, which is
        why the binarizer is necessary)
        '''
        x_matrix, y_matrix = UpsamplingData.split_x_y(training_data, binarizer)

        model = Sequential()
        model.add(Dense(units=100,
                        activation="relu",
                        input_dim=x_matrix.shape[1],
                        kernel_initializer='uniform'))
        model.add(Dense(units=50,
                        activation="relu"))
        model.add(Dense(units=y_matrix.shape[1],
                        activation="softmax"))
        model.compile(loss='binary_crossentropy', optimizer='adadelta',
                      metrics=['acc'])
        model.fit(x_matrix, y_matrix, epochs=100, batch_size=16)
        return model

    @staticmethod
    def evaluate_model(model: Sequential,
                       test_data: "DataFrame",
                       binarizer: Callable,
                       debinarizer: Callable) -> (np.matrix, dict, List[dict]):
        '''Evaluate model against test data and return a number of metrics.
        '''
        x_matrix, y_matrix = UpsamplingData.split_x_y(test_data,
                                                      binarizer)
        loss_and_metrics = model.evaluate(x_matrix, y_matrix, batch_size=128)
        print(loss_and_metrics)
        # y_pred = model.predict(x_matrix, batch_size=128)
        y_pred = model.predict_classes(x_matrix, batch_size=128)
        # convert y matrix to prediction class numbers to compare to y_pred
        ly_test = [np.argmax(x) for x in y_matrix]
        # create confusion matrix with predicted and actual labels
        confusion = skm.confusion_matrix(ly_test, y_pred)

        # print label and mismatch kind in detail
        mismatches = Classifier.get_mismatches(
            ly_test, y_pred, test_data, debinarizer)

        stats = {
            'accuracy': skm.accuracy_score(ly_test, y_pred),
            'precision': skm.precision_score(ly_test, y_pred,
                                             average="weighted"),
            'recall': skm.recall_score(ly_test, y_pred, average="weighted"),
            'f1': skm.f1_score(ly_test, y_pred, average="weighted")
        }
        return confusion, stats, mismatches

    @staticmethod
    def get_mismatches(true_labels: [int],
                       pred_labels: [int],
                       test_data: pd.DataFrame,
                       debinarizer: Callable) -> [dict]:
        '''Get dict of mismatched patients from classficiation result.
        '''
        mismatches = {}
        for i, y_vals in enumerate(zip(true_labels, pred_labels)):
            if y_vals[0] == y_vals[1]:
                continue
            label = test_data.iloc[i, :]["label"]
            mismatches[label] = {
                "true": debinarizer(y_vals[0], matrix=False),
                "predicted": debinarizer(y_vals[1], matrix=False)
            }
        return mismatches


def main():
    '''Simple binary classification
    '''
    data = UpsamplingData.from_file("../joined/joined_all.csv")
    Classifier(data)


if __name__ == '__main__':
    main()
