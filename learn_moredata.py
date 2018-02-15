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
from keras.utils import to_categorical,plot_model
from keras import regularizers
import seaborn
import functools
import matplotlib.pyplot as plt
from functools import reduce
'''
Flowcytometry classification methods
---
This file generates binary and multiclass classifications from the histogram data output
by flowSOM based preprocessing.

The input csv file contains relative numbers of events assigned to each cluster on the SOM
(self-organizing map) with additional columns containing label and group of each row, which
represents one fcs file.

The histograms are grouped according to groups and all groups are compared against each
other in the binary comparison. The larger group is always randomly sampled down to the
smaller group.

For neural network-based multiclass comparisons selected groups are also downsampled to the
size of the smallest group.
'''

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

def plot_learning_curve(history):
    plt.close('all')
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_acc.png')
    plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_loss.png')

def plot_size_acc(res):
    acc = [v['loss'][1] for v in res.values()]
    sen = [v['res']['Recall(Sensitivity)'] for v in res.values()]
    spec = [v['res']['Specificity'] for v in res.values()]
    x = list(res.keys())
    plt.close('all')
    plt.figure()
    seaborn.set()
    plt.ylim(0,1)
    plt.plot(x,acc)
    plt.plot(x,sen)
    plt.plot(x,spec)
    plt.title('Tube 1')
    plt.xlabel('Size of normal cohort')
    plt.legend(['accuracy','sensitivity','specificity'])
    plt.savefig('0000-tube1_size_all_plot.png')


def binary_ln(dataframes, positive, negative):
    input_df = dataframes['norm1'].set_index('label').join(
            dataframes['norm2'].set_index('label'),lsuffix='_a',rsuffix='_b',how='inner')
    assert ((input_df['group_a'] == input_df['group_b']).value_counts() == input_df.shape[0]).all(), \
            "Group labels are not identical for same ids"
    input_df = input_df.drop(['group_b','material_b','material_a'],axis=1)
    input_df = input_df.rename({'group_a':'group'},axis=1)

    # inputdf from single tubes

    input_df = old_df['norm1'].drop(['material','label'],axis=1)

    out = {}
    input_df = dataframes['norm1'].drop(['material','label'],axis=1)
    input_df = dataframes['norm2'].drop(['material','label'],axis=1)
    for nsize in range(100,3000,200):
        input_df = dataframes['norm2'].drop(['material','label'],axis=1)
        input_df = {k:input_df[input_df['group']==k] for k in input_df['group'].unique()}
        input_df['normal'] = input_df['normal'].sample(nsize)
        pos_dfs = {k:v for k,v in input_df.items() if k in positive}
        pos_df = functools.reduce(lambda x,y: x.append(y), pos_dfs.values())
        neg_dfs = {k:v for k,v in input_df.items() if k in negative}
        neg_df = functools.reduce(lambda x,y: x.append(y), neg_dfs.values())
        pos_tag = np.zeros((pos_df.shape[0],2))
        pos_tag[:,0] = 2
        neg_tag = np.zeros((neg_df.shape[0],2))
        neg_tag[:,1] = 1

        # num = int(min(pos_df.shape[0],neg_df.shape[0]) * 0.8)
        neg_df = shuffle(neg_df)#, random_state=0)
        # neg_df = neg_df.sample(min(pos_df.shape[0],neg_df.shape[0]))
        num = int(neg_df.shape[0] / 1.2)
        neg_train = neg_df.iloc[:num,:]
        neg_test = neg_df.iloc[num:,:]
        pos_df = shuffle(pos_df)#, random_state=0)
        # pos_df = pos_df.sample(min(pos_df.shape[0],neg_df.shape[0]))
        num = int(pos_df.shape[0] / 1.2)
        pos_train = pos_df.iloc[:num,:]
        pos_test = pos_df.iloc[num:,:]

        train_df = pandas.concat([neg_train,pos_train])
        train_df = train_df.drop(['group'],axis=1)
        # train_df -= train_df.min()
        # train_df /= train_df.max()
        train_tags = np.concatenate([neg_tag[:neg_train.shape[0],:],pos_tag[:pos_train.shape[0],:]])
        train_array = np.empty(train_df.shape)
        train_tag_list = np.empty(train_tags.shape)
        il = shuffle(range(train_df.shape[0]))
        for e,i in enumerate(il):
            train_array[e,:] = train_df.iloc[i,:].values
            train_tag_list[e,:] = train_tags[i,:]

        test_df = pandas.concat([neg_test,pos_test])
        test_df = test_df.drop(['group'],axis=1)
        # test_df -= test_df.min()
        # test_df /= test_df.max()
        test_tags = np.concatenate([neg_tag[:neg_test.shape[0],:],pos_tag[:pos_test.shape[0],:]])
        test_array = np.empty(test_df.shape)
        test_tag_list = np.empty(test_tags.shape)
        il = shuffle(range(test_df.shape[0]))
        for e,i in enumerate(il):
            test_array[e,:] = test_df.iloc[i,:].values
            test_tag_list[e,:] = test_tags[i,:]

        # auto class weights
        #class_weight = { 0: (pos_df.shape[0]+neg_df.shape[0]) / neg_df.shape[0]
        #        ,1:(pos_df.shape[0]+neg_df.shape[0])/pos_df.shape[0]}
        class_weight = {0:1,
                1:1 }

        model = Sequential()
        model.add(Dense(units=200, activation='relu', input_dim=train_array.shape[1],kernel_initializer='uniform'))
        model.add(Dense(units=200, activation='relu', input_dim=200))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['acc'])
        history = model.fit(train_array, train_tag_list, epochs=150, batch_size=64,class_weight=class_weight,validation_split=0.0)
        loss_and_metrics = model.evaluate(test_array, test_tag_list, batch_size=128)
        print(nsize)
        print(loss_and_metrics)
        y_pred = model.predict(test_array, batch_size=128)
        y_pred = model.predict_classes(test_array, batch_size=128)
        ly_test = [ int(x[1]) for x in test_tag_list ]
        cm = confusion_matrix(ly_test, y_pred)
        print(cm)
        precision = cm[0,0] / (cm[0,0]+cm[1,0])
        recall = cm[0,0] / (cm[0,1]+cm[0,0])
        f1 = 2/((1/precision)+(1/recall))
        specificity = cm[1,1]/(cm[1,1]+cm[1,0])
        npv = cm[1,1]/(cm[1,1]+cm[0,1])
        res = { 'Data':'more+new,400,tube2'
                ,'Precision(PPV)':precision
                ,'Recall(Sensitivity)':recall
                ,'F1 score':f1
                ,'Specificity':specificity
                ,'NPV':npv}
        out[nsize] = {'loss':loss_and_metrics
                ,'res':res}
        # write_results('tube2_400.txt', **res)
        # np.savetxt('confusion_400_tube2.txt',cm)
        # print("\n".join(["{} : {:.2}".format(k,v) for k,v in res.items()]))
        # plot_confusion_matrix(cm, ['patho','normal'],filename='confusion_bin_tube2_400.png')

def write_results(filename, **kwargs):
    res = ["{} : {}".format(k,v) for k,v in kwargs.items()]
    res = "\n".join(res)
    with open(filename, 'w') as fp:
        fp.write(res)


def run_neural_network(dataframes, groups, plotting=False):
    test_df = dataframes['norm1'].set_index('label').join(
            dataframes['norm2'].set_index('label'),lsuffix='_a',rsuffix='_b',how='inner')
    assert ((test_df['group_a'] == test_df['group_b']).value_counts() == test_df.shape[0]).all(), \
            "Group labels are not identical for same ids"

    test_df = test_df.drop(['group_b'],axis=1)
    test_df = test_df.rename({'group_a':'group'},axis=1)

    test_df = old_df['norm1']
    test_df = {k:test_df[test_df['group']==k] for k in test_df['group'].unique()}
    #select cohorts over 100 cases
    test_df = {k:v for k,v in test_df.items() if v.shape[0] >= 100}
    # test_df = {k:v for k,v in test_df.items() if k in groups}
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
    model.add(Dense(units=50, activation='relu', input_dim=data.shape[1],kernel_initializer='uniform'))#activity_regularizer=regularizers.l1(0.002)))#, kernel_initializer='uniform'))
    model.add(Dense(units=50, activation='relu', input_dim=50))#,activity_regularizer=regularizers.l1(0.002)))#, kernel_initializer='uniform'))
    #model.add(Dense(units=50, activation='relu', input_dim=50))#,kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(units=len(group.unique()), activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    if plotting:
        plot_model(model, to_file='neural_network.png')
        plot_model(model, show_shapes=True, to_file='network.png')
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)
    plot_learning_curve(history)
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
    print(cm)
    if plotting:
        plot_confusion_matrix(cm, groups)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, filename='confusion.png'):
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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename,dpi=300)

def main():
    csvs = {
             'norm1':'/data/ssdraid/Genetik/flowCat/400_Moredata_SET1_all_histos.csv'
            ,'meta1':'/data/ssdraid/Genetik/flowCat/400_Moredata_SET1_all_metas.csv'
            ,'norm2':'/data/ssdraid/Genetik/flowCat/400_Moredata_SET2_all_histos.csv'
            ,'meta2':'/data/ssdraid/Genetik/flowCat/400_Moredata_SET2_all_metas.csv'
            }
    csv_add = {
             'norm1':'/data/ssdraid/Genetik/flowCat/400_Newdata_SET1_all_histos.csv'
            ,'meta1':'/data/ssdraid/Genetik/flowCat/400_Newdata_SET1_all_metas.csv'
            ,'norm2':'/data/ssdraid/Genetik/flowCat/400_Newdata_SET2_all_histos.csv'
            ,'meta2':'/data/ssdraid/Genetik/flowCat/400_Newdata_SET2_all_metas.csv'
            }

    csv_other = {
            'norma':'/data/ssdraid/Genetik/flowCat/100_Newdata_SET1_all_histos.csv'
            ,'normb':'/data/ssdraid/Genetik/flowCat/100_Moredata_SET1_all_histos.csv'
            }
    csv_other = {k:pandas.read_csv(csv,delimiter=';') for k,csv in csv_other.items()}
    old_df = {'norm1':pandas.concat(csv_other.values())}

    PLOT_FOLDER = 'plots_full'
    LOGS_FOLDER = 'logs_full'

    plotting_data = False
    binary_comparisons = False
    plotting_binary = False
    neural_network = False
    plotting_neural = False

    binary_lympho = True

    if not os.path.exists(LOGS_FOLDER):
        os.mkdir(LOGS_FOLDER)
    if not os.path.exists(PLOT_FOLDER):
        os.mkdir(PLOT_FOLDER)

    dataframes_more = { k:pandas.read_csv(csv, delimiter=';') for k,csv in csvs.items() }
    dataframes_add = { k:pandas.read_csv(csv, delimiter=';') for k,csv in csv_add.items() }
    all_dataframes = {k:pandas.concat(v) for k,v in zip(dataframes_more.keys(),zip(dataframes_more.values(),dataframes_add.values()))}
    all_groups = { k:v['group'].unique() for k,v in all_dataframes.items() }
    if plotting_data:
        [ group_bar_plot(v,k,PLOT_FOLDER) for k,v in all_dataframes.items() ]
        [ data_histograms(v,all_groups[k],k,PLOT_FOLDER) for k,v in all_dataframes.items() ]

    if binary_comparisons:
        results = {k:binary_processing(v,all_groups[k],k) for k,v in all_dataframes.items()}
        if plotting_binary:
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

    if neural_network:
        #groups = ['Marginal','CLLPL','HZL','Mantel','CLL','normal']
        groups = ['CLL','normal','CLLPL','HZL','Mantel','Marginal','MBL']
        run_neural_network(all_dataframes, groups, plotting_neural)

    if binary_lympho:
        l = list(all_groups['norm1'])
        l.remove('normal')
        groups = {
                'negative':['normal']
                ,'positive': l
                }
        binary_ln(all_dataframes, groups['positive'],groups['negative'] )

if __name__ == '__main__':
    main()
