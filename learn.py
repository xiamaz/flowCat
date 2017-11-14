import os
import pandas
from sklearn import svm
from sklearn.utils import shuffle

os.chdir('/home/max/DREAM')
data = pandas.read_csv('matrix_output.csv', delimiter=';')

clf = svm.SVC(gamma=0.001, C=1000, kernel='rbf')

def svm_predict(df):
    pure_data = df.iloc[:,:-1]
    # labels = data.iloc[:,-1:]
    labels = df['label']
    clf.fit(pure_data[:-10], labels[:-10])
    truth = labels[-10:]
    p = clf.predict(pure_data[-10:])
    acc = 0;
    for i,s in enumerate(truth):
        if p[i] == s:
            acc += 1
    acc = acc / len(truth)
    print(acc)
    print(truth)

df = shuffle(data)
svm_predict(df)
