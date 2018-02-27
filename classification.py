'''
Neural network classification using keras
'''
from typing import Callable
from keras.models import Sequential
from keras.layers import Dense

from upsampling import UpsamplingData


class Classifier:
    '''Basic sequential dense classifier.'''

    def __init__(self, data: UpsamplingData):
        self.data = data

        self.train, self.test = self.data.get_test_train_split(ratio=0.8)

        self.model = self.create_sequential_model(self.train,
                                                  self.data.binarizer)

        self.evaluation = self.evaluate_model(self.model, self.test,
                                              self.data.binarizer)

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
                       binarizer: Callable) -> list:
        x_matrix, y_matrix = UpsamplingData.split_x_y(test_data,
                                                      binarizer)
        loss_and_metrics = model.evaluate(x_matrix, y_matrix, batch_size=128)
        print(loss_and_metrics)
        y_pred = model.predict(x_matrix, batch_size=128)
        print(y_pred)
        y_pred = model.predict_classes(x_matrix, batch_size=128)
        print(y_pred)
        # ly_test = [ int(x[1]) for x in test_tag_list ]
        # cm = confusion_matrix(ly_test, y_pred)
        # print(cm)
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
    data = UpsamplingData.from_file("../joined/cll_normal.csv")
    Classifier(data)


if __name__ == '__main__':
    main()
