'''
Neural network classification using keras
'''
import os
import logging
import json
import enum
from datetime import datetime
from typing import Callable, List, Union
from collections import defaultdict

import pandas as pd
import numpy as np
import sklearn.metrics as skm
from sklearn.tree import DecisionTreeClassifier

from keras import layers, models

from .upsampling import DataView
from .stamper import create_stamp
from . import plotting


def create_binarizers(names, create_matrix=True):
    '''Create and parse binary label descriptions
    '''
    def binarizer(factor):
        '''Create binary array from labels.
        '''
        value = names.index(factor)

        if create_matrix:
            bin_list = [0] * len(names)
            bin_list[value] = 1
            value = pd.Series(bin_list)

        return value

    def debinarizer(bin_array, matrix=True):
        '''Get label from binary array or position number.
        '''
        if matrix:
            val = np.argmax(bin_array)
        else:
            val = bin_array
        return names[val]

    return binarizer, debinarizer


def create_binarizer_from_data(
        col_data: pd.Series, matrix=True
) -> (Callable, Callable):
    '''Create binarizer and debinarizer from factors in a pandas series.
    '''
    group_names = sorted(list(col_data.unique()))
    binarizer, debinarizer = create_binarizers(
        group_names, create_matrix=matrix
    )
    return binarizer, debinarizer, group_names


def avg_dicts(dicts: [dict]) -> dict:
    """Average a dict"""
    avg_dict = defaultdict(float)
    for item in dicts:
        for key, value in item.items():
            avg_dict[key] += value
    avg_dict = {k: v / len(dicts) for k, v in avg_dict.items()}
    return avg_dict


def create_t2(predictions: pd.DataFrame) -> pd.DataFrame:
    arglist = predictions.apply(np.argsort, axis=1)
    vals = predictions.columns.to_series()[arglist.values[:, ::-1][:, :2]]
    return pd.DataFrame(vals, index=predictions.index)


def t2_accuracy_score(truth: list, t2pred: pd.DataFrame):
    t1 = t2pred.iloc[:, 0].values == truth.values
    t2 = t2pred.iloc[:, 1].values == truth.values
    tall = t1 | t2
    # quick and dirty accuracy calculation
    return sum(tall) / len(truth)


class BaseClassifier:
    """Basic classifier model."""
    def __init__(self):
        self.model = None
        self.binarizer = None
        self.debinarizer = None
        self.history = None
        self.groups = None

    def _fit_input(self, X, matrix=True):
        xdata, ydata = DataView.split_data_labels(X)
        x_matrix = xdata.values
        self.binarizer, self.debinarizer, self.groups = \
            create_binarizer_from_data(ydata, matrix=matrix)
        y_matrix = ydata.apply(self.binarizer).values
        return x_matrix, y_matrix

    def _pred_input(self, X):
        xdata, ydata = DataView.split_data_labels(X)
        return xdata.values


class Tree(BaseClassifier):
    """A normal tree."""

    def fit(self, X, *_):
        x_matrix, y_matrix = self._fit_input(X, matrix=False)

        self.model = DecisionTreeClassifier()
        self.model.fit(x_matrix, y_matrix)
        return self

    def predict(self, X, *_):
        x_matrix = self._pred_input(X)

        y_pred = self.model.predict(x_matrix)
        result = [self.debinarizer(x, matrix=False) for x in y_pred]
        return result


class BaseMultiTube(BaseClassifier):
    """Classification with individual tubes split into multiple
    inputs for neural network classification."""

    def _split_tubes(self, Xmatrix, length=100):
        """Split the matrix containing joined tubes into individual segments.
        Joining and splitting again is reasonable to keep the rowwise relation
        of individual cases."""
        _, width = Xmatrix.shape
        chunks = int(width / length)

        splits = np.hsplit(Xmatrix, chunks)
        return splits

    def predict_classes(self, X, *_):
        y_pred = self.predict(X).values
        if y_pred.shape[-1] > 1:
            preds = np.argmax(y_pred, axis=-1)
        else:
            preds = (y_pred > 0.5).astype('int32')

        result = [self.debinarizer(x, matrix=False) for x in preds]
        return result

    def _tube_pred_input(self, X):
        y_pred = self._split_tubes(self._pred_input(X))
        return y_pred

    def predict(self, X, *_):
        x_tubes = self._tube_pred_input(X)

        y_pred = self.model.predict(
            x_tubes, batch_size=128
        )
        y_pred_df = pd.DataFrame(y_pred, columns=self.groups, index=X["label"])
        return y_pred_df
        raise RuntimeError("Implement me")


class ConvNet(BaseMultiTube):
    """Basic testing conv net."""

    def _to_2d(self, Xmatrix):
        """Reshape case x histo data into a 2d map, similar to the
        initial SOM. Width needed for easier reshaping.
        """
        records, cols = Xmatrix.shape
        width = int(np.sqrt(cols))
        newshape = np.reshape(Xmatrix, (records, width, width, 1))
        print(newshape.shape)
        return newshape

    def fit(self, X, *_):
        x_matrix, y_matrix = self._fit_input(X)
        x_tubes = [self._to_2d(t) for t in self._split_tubes(x_matrix)]

        inputs = []
        parts = []
        for i in range(len(x_tubes)):
            tinput = layers.Input(shape=(10, 10, 1))
            inputs.append(tinput)

            inlayer = layers.Conv2D(
                32, kernel_size=(4, 4), strides=(1, 1),
                activation="relu",
                input_shape=(10, 10, 1)
            )(tinput)
            pooling = layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2)
            )(inlayer)
            flattened = layers.Flatten()(pooling)
            parts.append(flattened)

        concat = layers.concatenate(parts)
        output = layers.Dense(
            units=y_matrix.shape[1], activation="softmax"
        )(concat)

        model = models.Model(inputs=inputs, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['acc'])

        self.model = model

        self.history = self.model.fit(
            x_tubes, y_matrix, epochs=100, batch_size=32,
        )
        return self

    def predict(self, X, *_):
        x_tubes = [self._to_2d(t) for t in self._tube_pred_input(X)]

        y_pred = self.model.predict(
            x_tubes, batch_size=128
        )
        y_pred_df = pd.DataFrame(y_pred, columns=self.groups, index=X["label"])
        return y_pred_df


class MINet(BaseMultiTube):
    """Multiple input neural network with each tube as a separate input and
    later in network joining."""

    def fit(self, X, *_):
        tube_len = 100

        x_matrix, y_matrix = self._fit_input(X)
        x_tubes = self._split_tubes(x_matrix, length=tube_len)

        inputs = []
        parts = []
        for i in range(len(x_tubes)):
            tinput = layers.Input(shape=(tube_len, ))
            inputs.append(tinput)

            initial = layers.Dense(
                units=10, activation="elu", kernel_initializer='uniform'
            )(tinput)
            partend = layers.Dense(
                units=10, activation="elu"
            )(initial)

            parts.append(partend)

        concat = layers.concatenate(parts)

        output = layers.Dense(
            units=y_matrix.shape[1], activation="softmax"
        )(concat)

        model = models.Model(inputs=inputs, outputs=output)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['acc'],
        )
        class_weight = {
            i: 4.0 if g == "normal" else 1.0 for i, g in enumerate(self.groups)
        }
        self.model = model
        self.history = self.model.fit(
            x_tubes, y_matrix, epochs=100, batch_size=32,
            class_weight=class_weight
        )

        return self

    def predict(self, X, *_):
        x_tubes = self._tube_pred_input(X)
        y_pred = self.model.predict(
            x_tubes, batch_size=128
        )
        y_pred_df = pd.DataFrame(y_pred, columns=self.groups, index=X["label"])
        return y_pred_df


class NeuralNet(BaseClassifier):
    """Basic keras neural net."""
    def __init__(
            self, val_split: float = 0.0
    ):
        super().__init__()
        self.val_split = val_split

    @staticmethod
    def create_sequential(
            xshape: int, yshape: int,
            binarizer: Callable,
    ) -> models.Sequential:
        '''Create a sequential neural network with specified hidden layers.
        The input and output dimensions are inferred from the given
        data and labels. (Labels are converted to binary matrix, which is
        why the binarizer is necessary)
        '''
        model = models.Sequential()
        model.add(layers.Dense(
            units=10, activation="elu",
            input_dim=xshape, kernel_initializer='uniform'
        ))
        model.add(layers.Dense(
            units=10, activation="elu"
        ))
        model.add(layers.Dense(
            units=yshape, activation="softmax"
        ))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                      metrics=['acc'])
        return model

    def fit(self, X, *_):
        x_matrix, y_matrix = self._fit_input(X)

        self.model = self.create_sequential(
            x_matrix.shape[1], y_matrix.shape[1], self.binarizer
        )
        self.history = self.model.fit(
            x_matrix, y_matrix, epochs=100, batch_size=32,
            validation_split=self.val_split
        )
        return self

    def predict_classes(self, X, *_):
        x_matrix = self._pred_input(X)

        y_pred = self.model.predict_classes(
            x_matrix, batch_size=128
        )
        result = [self.debinarizer(x, matrix=False) for x in y_pred]
        return result

    def predict(self, X, *_):
        x_matrix = self._pred_input(X)

        y_pred = self.model.predict(
            x_matrix, batch_size=128
        )
        y_pred_df = pd.DataFrame(y_pred, columns=self.groups, index=X["label"])
        return y_pred_df


class MODELS(enum.Enum):
    """Enumeration for all current models."""
    NeuralNet = NeuralNet
    Tree = Tree
    ConvNet = ConvNet
    MINet = MINet

    @classmethod
    def from_str(cls, instr):
        return cls[instr]

    def get_model(self):
        return self.value


class Classifier:

    def __init__(
            self,
            data: DataView,
            output_path: str = "output/classification",
            name: str = "expname",
    ):
        # concatenate experiment name with a current timestamp
        self.name = name

        self._data = data

        self.output_path = os.path.join(output_path, self.name)
        os.makedirs(self.output_path, exist_ok=True)

        # log past experiment information in a dict
        self.past_experiments = []

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
            self.output_path, self.name + "_info.json")
        json.dump(dumpdata, open(dumppath, "w"), indent=4)

    def k_fold_validation(
            self,
            modelfunc: Callable,
            k_num: int = 5,
            **kwargs
    ) -> None:
        '''Do k-fold validation on data set.
        '''
        # put all output into a separate folder
        name_tag = "kfold"

        splits = self._data.k_fold_split(k_num)
        train_test_sets = [
            (pd.concat(splits[:i] + splits[i + 1:]), splits[i])
            for i in range(k_num)
        ]

        experiment_info = self.validation(
            modelfunc,
            train_test_sets,
            name_tag=name_tag,
            **kwargs
        )
        experiment_info["config_param"] = k_num

        self.past_experiments.append(experiment_info)

    def holdout_validation(
            self,
            modelfunc: Callable,
            ratio: float = 0.8,
            abs_num: int = None,
            save_weights: bool = True,
            val_split: float = 0.2,
            infiltration: float = 0.0,
    ) -> None:
        '''Simple holdout validation. If given abs_num, then each test cohort
        will contain the absolute number of cases.'''
        if abs_num:
            name_tag = "absolute"
            train, test = self._data.get_test_train_split(
                abs_num=abs_num, train_infiltration=infiltration
            )
        else:
            name_tag = "holdout"
            train, test = self._data.get_test_train_split(
                ratio=ratio, train_infiltration=infiltration
            )

        train_test_sets = [(train, test)]

        experiment_info = self.validation(
            modelfunc,
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
            modelfunc: Callable,
            train_test_sets: list,
            name_tag: str,
            save_weights: bool = False,
            save_individual_results: bool = False,
            plot_history: bool = False,
            val_split: float = 0.0,
    ) -> dict:
        '''Build models and create statistics for a list of train, test sets.
        '''
        eval_results = []
        prediction_dfs = []
        for i, (train, test) in enumerate(train_test_sets):
            model = modelfunc()
            model.fit(train)

            if save_weights:
                weight_file = os.path.join(
                    self.output_path, "weights_{}_{}.hdf5".format(name_tag, i)
                )
                model.model.save_weights(weight_file)

            if plot_history:
                plot_path = os.path.join(
                    self.output_path,
                    "training_history_{}_{}.png".format(name_tag, i)
                )
                plotting.plot_history(model.history, plot_path)
            training_stats = self.get_training_stats(model.history)

            predictions = model.predict_classes(test)
            test["prediction"] = predictions

            pred = model.predict(test)

            t2pred = create_t2(pred)

            pred["group"] = test["group"].values
            pred["infiltration"] = test["infiltration"].values
            prediction_dfs.append(pred)

            confusion, stat, mism = self.evaluate_model(
                predictions, test, t2pred
            )
            # add results for later batched interpretation
            eval_results.append((confusion, stat, mism, training_stats))
            # output individual results, if wanted
            if save_individual_results:
                self.generate_output(
                    confusion, mism, name_tag="_{}_{}".format(name_tag, i)
                )

        prediction_df = pd.concat(prediction_dfs)

        prediction_df.to_csv(
            os.path.join(
                self.output_path, name_tag + "_predictions" + ".csv"
            )
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
            "confusion": avg_confusion.tolist(),
            "groups": self._data.group_names,
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
            dendro_name = os.path.join(
                self.output_path, tagged_name + "_dendro.png")
            plotting.plot_confusion_matrix(
                confusion_data, self._data.group_names, normalize=True,
                filename=plot_name, dendroname=dendro_name
            )
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
    def evaluate_model(
            predictions: pd.DataFrame,
            test_data: pd.DataFrame,
            t2pred: pd.DataFrame,
    ) -> (np.matrix, dict, List[dict]):
        '''Evaluate model against test data and return a number of metrics.
        '''
        _, truth = DataView.split_data_labels(test_data)
        # create confusion matrix with predicted and actual labels
        confusion = skm.confusion_matrix(truth, predictions)

        mismatches = Classifier.get_mismatches(truth, predictions, test_data)

        stats = {
            'accuracy': skm.accuracy_score(
                truth, predictions
            ),
            't2accuracy': t2_accuracy_score(
                truth, t2pred
            ),
            'precision': skm.precision_score(
                truth, predictions, average="weighted"
            ),
            'recall': skm.recall_score(
                truth, predictions, average="weighted"
            ),
            'f1': skm.f1_score(
                truth, predictions, average="weighted"
            ),
        }
        return confusion, stats, mismatches

    @staticmethod
    def get_mismatches(
            true_labels: [int],
            pred_labels: [int],
            test_data: pd.DataFrame
    ) -> [dict]:
        '''Get dict of mismatched patients from classficiation result.
        '''
        mismatches = {}
        for i, y_vals in enumerate(zip(true_labels, pred_labels)):
            if y_vals[0] == y_vals[1]:
                continue
            label = test_data.iloc[i, :]["label"]
            mismatches[label] = {
                "true": y_vals[0],
                "predicted": y_vals[1],
            }
        return mismatches

    @staticmethod
    def get_training_stats(history: "History") -> dict:
        '''Get training stats from the training history.
        Currently only get get the last reported loss and accuracy.
        '''
        if history:
            loss = history.history["loss"]
            acc = history.history["acc"]
            return {"loss": loss[-1], "acc": acc[-1]}
        return {}
