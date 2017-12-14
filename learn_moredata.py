import pandas
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB

def run_optimization(df):
    # scale the data
    # datas -= datas.min()
    # datas /= datas.max()
    data, group, label = df.iloc[:,:-2], df['group'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(
            data, group, test_size=0.5, random_state=0)
    params = {
            'C' : list(range(1,500, 1))
            ,"gamma" : [0.1, 0.01, 0.001]
            ,'kernel' : ['rbf', 'linear']
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

def main():
    csvs = {
            'norm':'/home/max/DREAM/MOREDATA_all_histos.csv'
            ,'meta':'/home/max/DREAM/MOREDATA_all_metas.csv'
            }

    dataframes = { k:pandas.read_csv(csv, delimiter=';') for k,csv in csvs.items() }
    # normal selection
    normals = { k:df[df['group'] == 'normal'] for k,df in dataframes.items() }
    others = { k:df[df['group'] != 'normal'] for k,df in dataframes.items() }
    groups = { k:df['group'].unique() for k,df in others.items() }
    groups = {k:{ kk:others[k][others[k]['group'] == kk] for kk in l} for k, l in groups.items()}

    # comparisons against normal
    # for k,d in groups.items():
    for kk, df in groups['meta'].items():
        if (df.shape[0] < normals['meta'].shape[0] / 2):
            print("Skip {} because of size {}".format(kk, df.shape[0]))
            continue
        # add normal cohort to test group
        run_optimization(pandas.concat([df, normals['meta']]))



if __name__ == '__main__':
    main()
