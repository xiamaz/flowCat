'''
Neural network classification using keras
'''
import os
import logging
import json
from datetime import datetime
from typing import Callable, List, Union
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
        for key, value in item.items():
            avg_dict[key] += value
    avg_dict = {k: v/len(dicts) for k, v in avg_dict.items()}
    return avg_dict


class Classifier:
    '''Basic sequential dense classifier.'''

    def __init__(self,
                 data: UpsamplingData,
                 output_path: str = "output/classification",
                 name: str = "expname"):
        self._data = data
        os.makedirs(output_path, exist_ok=True)
        self.output_path = os.path.join(output_path, name)
        if os.path.exists(self.output_path):
            raise RuntimeError("{} already exists. Move or delete it.".format(
                self.output_path))
        os.makedirs(self.output_path)

        self.experiment_name = name

        # log past experiment information in a dict
        self.past_experiments = []

    def dump_experiment_info(self, note: str = "") -> None:
        '''Output all past run validations into a json file for later
        inference.'''
        dumpdata = {
            "experiment_name": self.experiment_name,
            "date": str(datetime.now()),
            "note": note,
            "group_names": self._data.group_names,
            "experiments": self.past_experiments
        }
        dumppath = os.path.join(
            self.output_path, self.experiment_name+"_info.json")
        json.dump(dumpdata, open(dumppath, "w"), indent=4)

    def k_fold_validation(self,
                          k_num: int = 5,
                          save_individual_results: bool = True) -> None:
        '''Do k-fold validation on data set.
        '''
        # put all output into a separate folder
        name_tag = "kfold"

        splits = self._data.k_fold_split(k_num)
        eval_results = []
        for i in range(k_num):
            # train on all other parts
            model = self.create_sequential_model(
                pd.concat(splits[i+1:] + splits[:i]),
                self._data.binarizer)
            # test using selected part i
            confusion, stat, mism = self.evaluate_model(
                model, splits[i], self._data.binarizer, self._data.debinarizer)
            # add results for later batched interpretation
            eval_results.append((confusion, stat, mism))
            # output individual results, if wanted
            if save_individual_results:
                self.generate_output(confusion, stat, mism,
                                     name_tag="_{}_{}".format(name_tag, i))

        avg_confusion = sum([t[0] for t in eval_results])
        avg_stats = avg_dicts([t[1] for t in eval_results])
        self.generate_output(
            confusion_data=avg_confusion,
            statistics=avg_stats,
            mismatches=None,
            name_tag=name_tag)

        experiment_info = {
            "setting": name_tag,
            "config_param": k_num,
            "splits": [s.shape[0] for s in splits]
        }
        self.past_experiments.append(experiment_info)

    def holdout_validation(self, ratio: float = 0.8,
                           abs_num: int = None,
                           save_weights: bool = True) -> None:
        '''Simple holdout validation. If given abs_num, then each test cohort
        will contain the absolute number of cases.'''
        if abs_num:
            name_tag = "absolute"
            train, test = self._data.get_test_train_split(abs_num=abs_num)
        else:
            name_tag = "holdout"
            train, test = self._data.get_test_train_split(ratio=ratio)

        model = self.create_sequential_model(
            train, self._data.binarizer)

        if save_weights:
            weight_file = os.path.join(self.output_path,
                                       "weights_" + name_tag + ".hdf5")
            model.save_weights(weight_file)

        confusion, stats, mismatches = self.evaluate_model(
            model, test, self._data.binarizer, self._data.debinarizer)

        self.generate_output(
            confusion_data=confusion,
            statistics=stats,
            mismatches=mismatches,
            name_tag=name_tag)

        experiment_info = {
            "setting": name_tag,
            "config_param": abs_num or ratio,
            "splits": [train.shape[0], test.shape[0]]
        }
        self.past_experiments.append(experiment_info)

    def generate_output(
            self,
            confusion_data: Union[np.matrix, None],
            statistics: Union[dict, None],
            mismatches: Union[dict, None],
            name_tag: str) -> None:
        '''Create confusion matrix and text statistics and save them to the
        output folder.
        '''
        tagged_name = "{}_{}".format(self.experiment_name, name_tag)

        # save confusion matrix
        if confusion_data is not None:
            logging.info("Create confusion matrix plot.")
            plot_name = os.path.join(
                self.output_path, tagged_name + "_confusion.png")
            plotting.plot_confusion_matrix(
                confusion_data, self._data.group_names,
                normalize=True, filename=plot_name)
        # save metrics of experiment
        if statistics is not None:
            logging.info("Saving statistics information to text file.")
            stat_file = os.path.join(self.output_path,
                                     tagged_name + "_statistics.txt")
            open(stat_file, "w").write(
                "{}\n====\n{}".format(tagged_name, statistics))
        # save list of mismatched cases
        if mismatches is not None:
            logging.info("Creating text list from mismatch information.")
            mism_file = os.path.join(self.output_path,
                                     tagged_name + "_mismatches.txt")
            mismatch_lines = [
                "{}: T<{}>|F<{}>\n".format(
                    label, value["true"], value["predicted"])
                for label, value in mismatches.items()
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
    files = [("../joined/joined_all.csv")]
    data = UpsamplingData.from_files(files)
    Classifier(data)


if __name__ == '__main__':
    main()
