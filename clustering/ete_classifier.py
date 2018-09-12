import numpy as np
import sklearn as sk

import keras
import fcsparser

from clustering import collection as cc


EVENT_COUNT = 10000


def create_ete_model(xdims, ydim):
    """Create end-to-end model. Use 1d convolutions."""
    i = keras.layers.Input(shape=xdims)

    x = i
    x = keras.layers.Conv1D(256, 1, strides=1, activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(l=0.001))(x)
    x = keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l=0.001))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(ydim, activation="softmax")(x)

    model = keras.models.Model(inputs=[i], outputs=x)
    return model


class FCSSequence(keras.utils.Sequence):
    def __init__(self, data, channels, binarizer, batch_size=32):
        self.data = np.random.permutation(data)
        self.batch_size = batch_size
        self.channels = channels
        self.binarizer = binarizer

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        xdat, ydat = get_merged_data(batch_data, channels=self.channels)
        mat_ydat = self.binarizer.transform(ydat)
        return xdat, mat_ydat


def get_merged_data(datas, channels):
    xdata = np.stack([
        d.get_merged_data(
            tubes=[1, 2], channels=channels
        ).sample(EVENT_COUNT * 2).fillna(0).sample(frac=1).values
        for d in datas
    ])
    ydata = np.array([d.group for d in datas])
    return xdata, ydata


def main():
    groups = ["CLL", "MBL", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"]
    lb = sk.preprocessing.LabelBinarizer().fit(groups)
    channels = [
        'FS INT LIN', 'SS INT LIN',
        'FMC7-FITC', 'CD10-PE', 'IgM-ECD',
        'CD79b-PC5.5', 'CD20-PC7', 'CD23-APC',
        'CD19-APCA750','CD5-PacBlue', 'CD45-KrOr',
        'Kappa-FITC', 'Lambda-PE', 'CD38-ECD',
        'CD25-PC5.5', 'CD11c-PC7', 'CD103-APC',
        'CD22-PacBlue'
    ]
    cases = cc.CaseCollection("tmp/CLL-9F", tubes=[1, 2])
    sel_cases = cases.create_view(num=1000, groups=groups, counts=EVENT_COUNT)

    train, test = sel_cases.create_split(num=180, stratify=True)

    model = create_ete_model((EVENT_COUNT * 2, len(channels)), len(groups))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

    train_seq = FCSSequence(train, channels, binarizer=lb, batch_size=128)
    model.fit_generator(train_seq, epochs=100, use_multiprocessing=True, workers=16)
    # model.fit(xtrain, mat_ytrain, batch_size=8, epochs=20)

    xtest, ytest = get_merged_data(test, channels=channels)
    mat_ytest = lb.transform(ytest)
    pred_mat = model.predict(xtest, batch_size=10)
    pred = lb.inverse_transform(pred_mat)
    # print(model.evaluate(xtest, mat_ytest, batch_size=10))
    confusion = sk.metrics.confusion_matrix(ytest, pred, groups,)
    print(confusion)

if __name__ == "__main__":
    main()
