import matplotlib
matplotlib.use('Agg')
import os
import pandas
import itertools
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer,label_binarize,LabelEncoder
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import seaborn
import matplotlib.pyplot as plt
from functools import reduce

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

def create_plot(plotf, path):
    plt.figure(dpi=300)
    seaborn.set(style='whitegrid')
    plt.tight_layout()
    plotf()
    plt.savefig(path)
    plt.close('all')

def group_bar_plot(df,k,pdir):
    occ = df.groupby('group').size()
    def plotfunc():
        ax = seaborn.barplot(x=occ.index,y=occ)
        ax.set(xlabel='Groups', ylabel='Number of files')
        loc,labels = plt.xticks()
        ax.set_xticklabels(labels,rotation=30)
    create_plot(plotfunc,'{}/num_overview_{}.png'.format(pdir,k))

def data_histograms(df,groups,k,pdir):
    plotdir = os.path.join(pdir,k)
    num = df.shape[1] - 2
    x1 = np.array(range(0,num))
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    for g in groups:
        d = df[df['group'] == g]
        data, group, label = d.iloc[:,:-2], d['group'], d['label']
        if 'meta' in k:
            def plotfunc():
                plt.title('{} {} cell cluster distribution'.format(k,g))
                # ax = seaborn.violinplot(data=data, palette="Set3", bw=.2, cut=1, linewidth=1)
                ax = seaborn.swarmplot(data=data)
                ax.set(xlabel='FlowSOM clusters', ylabel='relative percentage of cells')
                seaborn.despine(left=True, bottom=True)
                plt.ylim(0,1)
        else:
            x1 = list(range(1,data.shape[1]+1))
            def plotfunc():
                plt.title('{} {} cell cluster distribution'.format(k,g))
                for i in range(0,data.shape[0]):
                    y = np.array(data.iloc[i,:],dtype='float')
                    plt.scatter(x1, y)
                ax = plt.gca()
                ax.set(xlabel='FlowSOM clusters', ylabel='relative percentage of cells')
                seaborn.despine(left=True, bottom=True)
                plt.ylim(0,1)
        create_plot(plotfunc, os.path.join(plotdir,g))

def run_neural_network(df, outfile_nn):
    data,group,label = df.iloc[:,:-2],df['group'],df['label']
    X_train, X_test, y_train, y_test = train_test_split(
            data, group, test_size=0.5, random_state=0)
    model = Sequential()
    model.add(units=64, activation='relu', input_dim=data.shape[1])
    model.add(units=len(label.unique()), activation='softmax')
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
    print(loss_and_metrics,file=outfile_nn)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.close('all')
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion.png',dpi=300)

def main():
    # csvs = {
    #         'norm1':'/home/max/DREAM/flowCat/4_MOREDATA_SET1_all_histos.csv'
    #         ,'meta1':'/home/max/DREAM/flowCat/4_MOREDATA_SET1_all_metas.csv'
    #         ,'norm2':'/home/max/DREAM/flowCat/4_MOREDATA_SET2_all_histos.csv'
    #         ,'meta2':'/home/max/DREAM/flowCat/4_MOREDATA_SET2_all_metas.csv'
    #         }
    csvs = {
            'norm1':'/home/max/DREAM/flowCat/1_MOREDATA_SET1_all_histos.csv'
            ,'meta1':'/home/max/DREAM/flowCat/1_MOREDATA_SET1_all_metas.csv'
            ,'norm2':'/home/max/DREAM/flowCat/2_MOREDATA_SET2_all_histos.csv'
            ,'meta2':'/home/max/DREAM/flowCat/2_MOREDATA_SET2_all_metas.csv'
            }
    PLOT_FOLDER = 'plots_even'
    LOGS_FOLDER = 'logs_more'

    if not os.path.exists(LOGS_FOLDER):
        os.mkdir(LOGS_FOLDER)
    if not os.path.exists(PLOT_FOLDER):
        os.mkdir(PLOT_FOLDER)

    dataframes = { k:pandas.read_csv(csv, delimiter=';') for k,csv in csvs.items() }
    all_groups = { k:v['group'].unique() for k,v in dataframes.items() }
    [ group_bar_plot(v,k,PLOT_FOLDER) for k,v in dataframes.items() ]
    [ data_histograms(v,all_groups[k],k,PLOT_FOLDER) for k,v in dataframes.items() ]


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

    test_df = dataframes['norm1'].set_index('label').join(
            dataframes['norm2'].set_index('label'),lsuffix='_a',rsuffix='_b',how='inner')
    assert ((test_df['group_a'] == test_df['group_b']).value_counts() == test_df.shape[0]).all(), \
            "Group labels are not identical for same ids"

    test_df = test_df.drop(['group_b'],axis=1)
    test_df = test_df.rename({'group_a':'group'},axis=1)

    test_df = {k:test_df[test_df['group']==k] for k in test_df['group'].unique()}
    # groups = ['Marginal','CLLPL','HZL','Mantel','CLL','normal']
    groups = ['CLL', 'normal','CLLPL','HZL', 'Mantel', 'Marginal','MBL']
    test_df = {k:v for k,v in test_df.items() if k in groups}
    minsize = min(map(lambda x: x.shape[0], test_df.values()))
    d = None
    for k,v in test_df.items():
        if d is None:
            d = shuffle(v, random_state=0, n_samples=minsize)
        else:
            d = d.append(shuffle(v, random_state=0, n_samples=minsize))

    data,group = d.drop(['group'],axis=1),d['group']
    lb = LabelBinarizer()
    lb.fit(groups)
    bgroup = lb.transform(group)

    X_train, X_test, y_train, y_test = train_test_split(
            data, bgroup, test_size=0.2, random_state=0)
    X_train -= X_train.min()
    X_train /= X_train.max()
    X_train = X_train.values
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=data.shape[1]))
    model.add(Dense(units=50, activation='relu', input_dim=50))
    model.add(Dense(units=50, activation='relu', input_dim=50))
    model.add(Dense(units=len(group.unique()), activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=10)
    X_test -= X_test.min()
    X_test /= X_test.max()
    X_test = X_test.values
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
    print(loss_and_metrics)
    y_pred = model.predict(X_test, batch_size=128)
    y_pred = model.predict_classes(X_test, batch_size=128)
    le = LabelEncoder()
    le.fit(groups)
    ly_pred = le.inverse_transform(y_pred)
    ly_test = lb.inverse_transform(y_test)
    cm = confusion_matrix(ly_test, ly_pred,labels=groups)
    plot_confusion_matrix(cm, groups)


if __name__ == '__main__':
    main()
