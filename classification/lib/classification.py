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

from lib.upsampling import DataView
from lib.stamper import create_stamp
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
                 data: DataView,
                 output_path: str = "output/classification",
                 name: str = "expname"):
        # concatenate experiment name with a current timestamp
        self.name = name

        self._data = data

        self.output_path = os.path.join(output_path, self.name)
        os.makedirs(self.output_path, exist_ok=True)

        # log past experiment information in a dict
        self.past_experiments = []

    def get_results(self) -> dict:
        '''Get results from previously run experiments.'''
        return self.past_experiments

    def dump_experiment_info(
            self,
            note: str = "",
            cmd: str = ""
    ) -> None:
        '''Output all past run validations into a json file for later
        inference.'''
        dumpdata = {
            "experiment_name": self.name,
            "date": str(datetime.now()),
            "note": note,
            "command": cmd,
            "group_names": self._data.group_names,
            "experiments": self.past_experiments
        }
        dumppath = os.path.join(
            self.output_path, self.name+"_info.json")
        json.dump(dumpdata, open(dumppath, "w"), indent=4)

    def k_fold_validation(
            self,
            k_num: int = 5,
            **kwargs
    ) -> None:
        '''Do k-fold validation on data set.
        '''
        # put all output into a separate folder
        name_tag = "kfold"

        splits = self._data.k_fold_split(k_num)
        train_test_sets = [
            (pd.concat(splits[:i] + splits[i+1:]), splits[i])
            for i in range(k_num)
        ]

        experiment_info = self.validation(
            train_test_sets,
            name_tag=name_tag,
            **kwargs
        )
        experiment_info["config_param"] = k_num

        self.past_experiments.append(experiment_info)

    def holdout_validation(
            self,
            ratio: float = 0.8,
            abs_num: int = None,
            save_weights: bool = True,
            val_split: float = 0.2
    ) -> None:
        '''Simple holdout validation. If given abs_num, then each test cohort
        will contain the absolute number of cases.'''
        if abs_num:
            name_tag = "absolute"
            train, test = self._data.get_test_train_split(abs_num=abs_num)
        else:
            name_tag = "holdout"
            train, test = self._data.get_test_train_split(ratio=ratio)

        train_test_sets = [(train, test)]

        experiment_info = self.validation(
            train_test_sets, name_tag=name_tag,
            save_weights=save_weights,
            save_individual_results=True,
            plot_history=True,
            val_split=val_split
        )

        experiment_info["config_param"] = abs_num or ratio
        self.past_experiments.append(experiment_info)

    def validation(
            self,
            train_test_sets: list,
            name_tag: str,
            save_weights: bool = False,
            save_individual_results: bool = False,
            plot_history: bool = False,
            val_split: float = 0.0
    ) -> dict:
        '''Build models and create statistics for a list of train, test sets.
        '''
        eval_results = []
        for i, (train, test) in enumerate(train_test_sets):
            model, history = self.create_sequential_model(
                train, self._data.binarizer,
                val_split=val_split
            )
            if save_weights:
                weight_file = os.path.join(
                    self.output_path, "weights_{}_{}.hdf5".format(name_tag, i)
                )
                model.save_weights(weight_file)

            if plot_history:
                plot_path = os.path.join(
                    self.output_path,
                    "training_history_{}_{}.png".format(name_tag, i)
                )
                plotting.plot_history(history, plot_path)
            training_stats = self.get_training_stats(history)

            confusion, stat, mism = self.evaluate_model(
                model, test, self._data.binarizer, self._data.debinarizer
            )
            # add results for later batched interpretation
            eval_results.append((confusion, stat, mism, training_stats))
            # output individual results, if wanted
            if save_individual_results:
                self.generate_output(
                    confusion, mism, name_tag="_{}_{}".format(name_tag, i)
                )

        avg_confusion = sum([t[0] for t in eval_results])
        avg_stats = avg_dicts([t[1] for t in eval_results])
        avg_training = avg_dicts([t[3] for t in eval_results])

        self.generate_output(
            confusion_data=avg_confusion,
            mismatches=None,
            name_tag=name_tag
        )

        experiment_info = {
            "setting": name_tag,
            "splits": [
                dict(
                    zip(
                        ["train", "test"],
                        [
                            {
                                "size": tv.shape[0],
                                "groups": tv.groupby("group").size().to_dict(),
                            } for tv in t
                        ]
                    )
                )
                for t in train_test_sets
            ],
            "avg_training": avg_training,
            "avg_result": avg_stats,
            "individual": [
                {
                    "result": stat,
                    "mismatches": mismatches,
                    "training": training
                }
                for _, stat, mismatches, training in eval_results
            ]
        }
        return experiment_info

    def generate_output(
            self,
            confusion_data: Union[np.matrix, None],
            mismatches: Union[dict, None],
            name_tag: str
    ) -> None:
        '''Create confusion matrix and text statistics and save them to the
        output folder.
        '''
        tagged_name = "{}_{}".format(self.name, name_tag)

        # save confusion matrix
        if confusion_data is not None:
            logging.info("Create confusion matrix plot.")
            plot_name = os.path.join(
                self.output_path, tagged_name + "_confusion.png")
            plotting.plot_confusion_matrix(
                confusion_data, self._data.group_names,
                normalize=True, filename=plot_name)
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
    def create_sequential_model(
            training_data: "DataFrame",
            binarizer: Callable,
            val_split: float = 0.0
    ) -> Sequential:
        '''Create a sequential neural network with specified hidden layers.
        The input and output dimensions are inferred from the given
        data and labels. (Labels are converted to binary matrix, which is
        why the binarizer is necessary)
        '''
        x_matrix, y_matrix = DataView.split_x_y(training_data, binarizer)

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
        history = model.fit(
            x_matrix, y_matrix, epochs=200, batch_size=16,
            validation_split=val_split
        )
        return model, history

    @staticmethod
    def evaluate_model(
            model: Sequential,
            test_data: "DataFrame",
            binarizer: Callable,
            debinarizer: Callable
    ) -> (np.matrix, dict, List[dict]):
        '''Evaluate model against test data and return a number of metrics.
        '''
        x_matrix, y_matrix = DataView.split_x_y(test_data, binarizer)
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

    @staticmethod
    def get_training_stats(history: "History") -> dict:
        '''Get training stats from the training history.
        Currently only get get the last reported loss and accuracy.
        '''
        loss = history.history["loss"]
        acc = history.history["acc"]
        return {"loss": loss[-1], "acc": acc[-1]}
