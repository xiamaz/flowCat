import pandas
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


def split_frame_cross(df, splits=5):
    return np.array_split(df.iloc[:,:-1], splits), np.array_split(df['label'], splits)
def split_frame(df):
    return df.iloc[:,:-1], df['label']

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

def scale_datas(data):
    data -= data.min()
    data /= data.max()
    return data

def main():
    data_csv = '/home/max/DREAM/matrix_meta_output_new3.csv'
    splits = 10

    dataframe = pandas.read_csv(data_csv, delimiter=';')
    # datas, labels = split_frame_cross(dataframe, splits=splits)
    datas, labels = split_frame(dataframe)
    datas = scale_datas(datas)
    X_train, X_test, y_train, y_test = train_test_split(
            datas, labels, test_size=0.5, random_state=0)

    params = {
            'C' : list(range(1,1000, 1))
            ,"gamma" : [0.1, 0.01, 0.001]
            ,'kernel' : ['linear', 'rbf', 'poly']
            }
    grid_search = GridSearchCV(svm.SVC(), params, n_jobs=8)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    # print("Grid scores on development set:\n\n")
    # means = grid_search.cv_results_['mean_test_score']
    # stds = grid_search.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))

    # scores = svm_predict(clf, datas, labels, splits=splits)
    # print(scores)

if __name__ == '__main__':
    main()
