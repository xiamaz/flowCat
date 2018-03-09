'''
Neural network classification using keras
'''
from typing import Callable

import numpy as np
import sklearn.metrics
from keras.models import Sequential
from keras.layers import Dense

from upsampling import UpsamplingData
from lib import plotting


class Classifier:
    '''Basic sequential dense classifier.'''

    def __init__(self, data: UpsamplingData):
        self.data = data

        self.train, self.test = self.data.get_test_train_split(ratio=0.8)

        self.model = self.create_sequential_model(self.train,
                                                  self.data.binarizer)

        self.evaluation = self.evaluate_model(self.model, self.test,
                                              self.data.binarizer,
                                              self.data.debinarizer,
                                              self.data.group_names)

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
                       group_names: [str]) -> None:
        x_matrix, y_matrix = UpsamplingData.split_x_y(test_data,
                                                      binarizer)
        loss_and_metrics = model.evaluate(x_matrix, y_matrix, batch_size=128)
        print(loss_and_metrics)
        # y_pred = model.predict(x_matrix, batch_size=128)
        y_pred = model.predict_classes(x_matrix, batch_size=128)
        # convert y matrix to prediction class numbers to compare to y_pred
        ly_test = [np.argmax(x) for x in y_matrix]
        # create confusion matrix with predicted and actual labels
        confusion = sklearn.metrics.confusion_matrix(ly_test, y_pred)
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
                                       filename="all_fusion.png")
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


def main():
    '''Simple binary classification
    '''
    data = UpsamplingData.from_file("../joined/joined_all.csv")
    Classifier(data)


if __name__ == '__main__':
    main()
