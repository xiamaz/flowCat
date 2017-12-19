import matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.use('Agg')
import os
import pandas
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from keras.models import Sequential
from keras.layers import Dense
import seaborn
import matplotlib.pyplot as plt

outfile = None

def run_optimization(df): # scale the data
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
    print(grid_search.best_params_,file=outfile)
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred),file=outfile)
    return precision_recall_fscore_support(y_test, y_pred, average='weighted')

def run_neural_network(df, outfile_nn):
    data,group,label = df.iloc[:,:,:-2],df['group'],df['label']
    X_train, X_test, y_train, y_test = train_test_split(
            data, group, test_size=0.5, random_state=0)
    model = Sequential()
    model.add(units=64, activation='relu', input_dim=data.shape[1])
    model.add(units=len(label.unique()), activation='softmax')
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
    print(loss_and_metrics,file=outfile_nn)

def data_histograms(df,groups,k):
    seaborn.set()
    num = df.shape[1] - 2
    x1 = np.array(range(0,num))
    if not os.path.exists('plots_new/{}'.format(k)):
        os.makedirs('plots_new/{}'.format(k))
    for g in groups:
        d = df[df['group'] == g]
        data, group, label = d.iloc[:,:-2], d['group'], d['label']
        plt.figure()
        for i in range(0,data.shape[0]):
            y = np.array(data.iloc[i,:],dtype='float')
            plt.plot(x1, y)
        plt.savefig('plots_new/{}/{}.png'.format(k,g))

def binary_processing(dfs, groups, meta):
    if not os.path.exists('logs/{}'.format(meta)):
        os.makedirs('logs/{}'.format(meta))
    result_dict = {
            'precision':pandas.DataFrame(0, index=groups, columns=groups)
            ,'recall':pandas.DataFrame(0, index=groups, columns=groups)
            ,'f1':pandas.DataFrame(0,index=groups,columns=groups)
            }
    for g in groups:
        outfile = open("logs/{}/{}.txt".format(meta,g), 'w')
        normals = dfs[dfs['group']==g]
        others = dfs[dfs['group']!=g]
        groups = {g:others[others['group']==g] for g in others['group'].unique()}
        if normals.shape[0] < 20:
            print("Skip {} entire group because of small size {}".format(g, normals.shape[0]),file=outfile)
            continue
        for kk, df in groups.items():
            if df.shape[0] < 20:
                print("Skip {} entire group because of small size {}".format(g, normals.shape[0]),file=outfile)
                continue
            t1 = shuffle(normals, random_state=0, n_samples=min(normals.shape[0],df.shape[0]))
            t2 = shuffle(df, random_state=0, n_samples=min(normals.shape[0],df.shape[0]))
            result = run_optimization(pandas.concat([t1, t2]))
            result_dict['precision'].loc[g,kk] = result[0]
            result_dict['recall'].loc[g,kk] = result[1]
            result_dict['f1'].loc[g,kk] = result[2]
        outfile.close()
    return result_dict

def group_bar_plot(df,k):
    occ = df.groupby('group').size()
    plt.figure()
    ax = seaborn.barplot(x=occ.index,y=occ)
    ax.set(xlabel='Groups', ylabel='Number of files')
    loc,labels = plt.xticks()
    ax.set_xticklabels(labels,rotation=30)
    plt.savefig('plots_new/num_overview_{}.png'.format(k), dpi=300)


def main():
    csvs = {
            'norm1':'/home/max/DREAM/flowCat/4_MOREDATA_SET1_all_histos.csv'
            ,'meta1':'/home/max/DREAM/flowCat/4_MOREDATA_SET1_all_metas.csv'
            ,'norm2':'/home/max/DREAM/flowCat/4_MOREDATA_SET2_all_histos.csv'
            ,'meta2':'/home/max/DREAM/flowCat/4_MOREDATA_SET2_all_metas.csv'
            }

    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('plots_new'):
        os.mkdir('plots_new')

    dataframes = { k:pandas.read_csv(csv, delimiter=';') for k,csv in csvs.items() }
    all_groups = { k:v['group'].unique() for k,v in dataframes.items() }
    [ group_bar_plot(v,k) for k,v in dataframes.items() ]
    [ data_histograms(v,all_groups[k],k) for k,v in dataframes.items() ]
    plt.close('all')


    results = {k:binary_processing(v,all_groups[k],k) for k,v in dataframes.items()}
    for k,v in results.items():
        for kk,vv in v.items():
            plt.figure()
            plt.subplots(figsize=(10,8))
            plt.title("{} {} heatmap".format(k,kk))
            mask = np.zeros_like(vv)
            mask[np.triu_indices_from(mask)] = True
            seaborn.heatmap(vv, cmap='YlGnBu', annot=True, fmt='.2f',mask=mask)
            plt.savefig('plots_new/heatmap_{}_{}.png'.format(k,kk))
        plt.close('all')

if __name__ == '__main__':
    main()
