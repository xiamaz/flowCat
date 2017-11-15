import pandas
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle


def split_frame(df, splits=5):
    return np.array_split(df.iloc[:,:-1], splits), np.array_split(df['label'], splits)

def svm_predict(clf, data, labels, splits=5):
    scores = []
    for k in range(splits):
        x_train = list(data)
        x_test = x_train.pop(k)
        x_train = np.concatenate(x_train)
        y_train = list(labels)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        scores.append(clf.fit(x_train, y_train).score(x_test, y_test))
    return scores

def main():
    data_csv = 'matrix_output.csv'
    splits = 10

    dataframe = pandas.read_csv(data_csv, delimiter=';')
    datas, labels = split_frame(dataframe, splits=splits)

    clf = svm.SVC(gamma=0.001, C=100, kernel='linear')
    scores = svm_predict(clf, datas, labels, splits=splits)
    print(scores)

if __name__ == '__main__':
    main()
