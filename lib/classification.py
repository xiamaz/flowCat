'''
Neural network classification using keras
'''
from typing import Callable

import pandas as pd
import numpy as np
import sklearn.metrics as skm
from keras.models import Sequential
from keras.layers import Dense

from lib.upsampling import UpsamplingData
from lib import plotting


class Classifier:
    '''Basic sequential dense classifier.'''

    def __init__(self, data: UpsamplingData):
        self.data = data

    def k_fold_validation(self, name: str="expname", k_num: int=5):
        '''Do k-fold validation on data set.
        '''
        splits = self.data.k_fold_split(k_num)
        evals = []
        for i in range(k_num):
            test = splits[i]
            train = pd.concat(splits[i+1:] + splits[:i])
            model = self.create_sequential_model(train, self.data.binarizer)
            stats = self.evaluate_model(
                model, test, self.data.binarizer, self.data.debinarizer,
                self.data.group_names, filename=name + "_kfold_{}".format(i))
            evals.append(stats)
        averaged = {}
        for eval_data in evals:
            for key, value in eval_data.items():
                if key not in averaged:
                    averaged[key] = value
                else:
                    averaged[key] += value
        averaged = {key: value/len(evals) for key, value in averaged.items()}
        print("Averaged k-fold values:")
        print(averaged)
        return averaged

    def holdout_validation(self, name: str="expname", ratio: float=0.8):
        '''Simple holdout validation.'''
        self.train, self.test = self.data.get_test_train_split(ratio=ratio)

        self.model = self.create_sequential_model(self.train,
                                                  self.data.binarizer)

        self.evaluation = self.evaluate_model(self.model, self.test,
                                              self.data.binarizer,
                                              self.data.debinarizer,
                                              self.data.group_names,
                                              filename=name + "_holdout")

    def absolute_validation(self, name="expname", abs_num=20):
        '''Validate on fixed size test.'''
        self.train, self.test = self.data.get_test_train_split(abs_num=abs_num)

        self.model = self.create_sequential_model(self.train,
                                                  self.data.binarizer)

        self.evaluation = self.evaluate_model(self.model, self.test,
                                              self.data.binarizer,
                                              self.data.debinarizer,
                                              self.data.group_names,
                                              filename=name + "_absolute")

    @staticmethod
    def create_sequential_model(training_data: "DataFrame",
                                binarizer: Callable) -> Sequential:
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
    def evaluate_model(model: Sequential, test_data: "DataFrame",
                       binarizer: Callable,
                       debinarizer: Callable,
                       group_names: [str],
                       filename: str) -> None:
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
        print(group_names)
        print(confusion)

        # print label and mismatch kind in detail
        print("All mismatches")
        for i, y_vals in enumerate(zip(ly_test, y_pred)):
            if y_vals[0] == y_vals[1]:
                continue
            label = test_data.iloc[i, :]["label"]
            print("{}:{}|{}".format(label,
                                    debinarizer(y_vals[0], matrix=False),
                                    debinarizer(y_vals[1], matrix=False)))

        plotting.plot_confusion_matrix(confusion, group_names,
                                       normalize=False,
                                       filename=filename + "_confusion.png")

        stats = "Loss and metrics\n" + repr(loss_and_metrics) + "\n"
        accuracy = skm.accuracy_score(ly_test, y_pred)
        stats += "Accuracy\n" + repr(accuracy) + "\n"
        # avg_precision = skm.average_precision_score(ly_test, y_pred,
        #                                            average="weighted")
        # stats += "Average precision\n" + repr(avg_precision) + "\n"
        precision = skm.precision_score(ly_test, y_pred, average="weighted")
        stats += "Weighted precision\n" + repr(precision) + "\n"
        recall = skm.recall_score(ly_test, y_pred, average="weighted")
        stats += "Weighted recall\n" + repr(recall) + "\n"
        open(filename + "_log.txt", "w").write(stats)
        # precision = cm[0,0] / (cm[0,0]+cm[1,0])
        # recall = cm[0,0] / (cm[0,1]+cm[0,0])
        # f1 = 2/((1/precision)+(1/recall))
        # specificity = cm[1,1]/(cm[1,1]+cm[1,0])
        # npv = cm[1,1]/(cm[1,1]+cm[0,1])
        # res = { 'Data':'more+new,400,tube2'
        #         ,'Precision(PPV)':precision
        #         ,'Recall(Sensitivity)':recall
        #         ,'F1 score':f1
        #         ,'Specificity':specificity
        #         ,'NPV':npv}
        # out[nsize] = {'loss':loss_and_metrics
        #         ,'res':res}

        stats = {
            # 'loss_metrics': loss_and_metrics,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }
        return stats


def main():
    '''Simple binary classification
    '''
    data = UpsamplingData.from_file("../joined/joined_all.csv")
    Classifier(data)


if __name__ == '__main__':
    main()
