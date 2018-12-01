import multiprocessing
import hashlib

import numpy as np
import sklearn as sk

import keras
import fcsparser

from clustering import collection as cc


EVENT_COUNT = 200
CHANNELS = [
    'FS INT LIN', 'SS INT LIN',
    'FMC7-FITC', 'CD10-PE', 'IgM-ECD',
    'CD79b-PC5.5', 'CD20-PC7', 'CD23-APC',
    'CD19-APCA750','CD5-PacBlue', 'CD45-KrOr',
    'Kappa-FITC', 'Lambda-PE', 'CD38-ECD',
    'CD25-PC5.5', 'CD11c-PC7', 'CD103-APC',
    'CD22-PacBlue'
]


def create_ete_model(xdims, ydim, num_inputs=2):
    """Create end-to-end model. Use 1d convolutions."""
    inputs = []
    tubes = []
    for num in range(num_inputs):
        i = keras.layers.Input(shape=xdims[num])
        inputs.append(i)

        x = i
        xa = keras.layers.Conv1D(128, 1, strides=1, activation="relu")(x)
        xa = keras.layers.GlobalAveragePooling1D()(xa)
        # xa = keras.layers.Dropout(0.2)(xa)
        # x = xa
        xb = keras.layers.Conv1D(32, 1, strides=1, activation="relu")(x)
        xb = keras.layers.GlobalMaxPool1D()(xb)
        xb = keras.layers.BatchNormalization()(xb)
        # xb = keras.layers.Dropout(0.2)(xb)
        x = keras.layers.concatenate([xa, xb])

        # x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(l=0.001))(x)
        # x = keras.layers.Dropout(0.2)(x)
        tubes.append(x)

    x = keras.layers.concatenate(tubes)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(ydim, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


class FCSSequence(keras.utils.Sequence):
    def __init__(self, data, channels, binarizer, batch_size=32):
        self.data = np.random.permutation(data)
        self.batch_size = batch_size
        self.channels = channels
        self.binarizer = binarizer
        self._cache = {}

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        xdat, ydat = get_merged_data(batch_data, channels=self.channels)
        mat_ydat = self.binarizer.transform(ydat)
        self._cache[idx] = (xdat, mat_ydat)
        return xdat, mat_ydat

def get_mat1(data):
    arrdata = data.get_merged_data(
        tubes=[1], channels=None
    ).sample(EVENT_COUNT).fillna(0).sample(frac=1).values
    return arrdata

def get_mat2(data):
    arrdata = data.get_merged_data(
        tubes=[2], channels=None
    ).sample(EVENT_COUNT).fillna(0).sample(frac=1).values
    return arrdata


def get_merged_data(datas):
    with multiprocessing.Pool(8) as p:
        xdata1_list = p.map(get_mat1, datas)
        xdata1 = np.stack(xdata1_list)
        xdata2_list = p.map(get_mat2, datas)
        xdata2 = np.stack(xdata2_list)
    # xdata = np.stack([
    #     d.get_merged_data(
    #         tubes=[1, 2], channels=channels
    #     ).sample(EVENT_COUNT * 2).fillna(0).sample(frac=1).values
    #     for d in datas
    # ])
        ydata = np.array([d.group for d in datas])
        return [xdata1, xdata2], ydata


def main():
    groups = ["CLL", "MBL", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"]
    mapped_groups = ["CM", "MP", "LM", "FL", "HCL", "normal"]
    group_map = {
        "CLL": "CM",
        "MBL": "CM",
        "MZL": "LM",
        "LPL": "LM",
        "MCL": "MP",
        "PL": "MP",
    }
    map_groups = np.vectorize(lambda d: group_map.get(d, d))
    # channels = CHANNELS

    lb = sk.preprocessing.LabelBinarizer().fit(mapped_groups)
    cases = cc.CaseCollection("/home/zhao/tmp/CLL-9F", tubes=[1, 2])
    sel_cases = cases.create_view(num=1000, groups=groups, counts=EVENT_COUNT)

    train, test = sel_cases.create_split(num=0.2, stratify=True)
    xtrain, ytrain = get_merged_data(train.data)
    mapped_ytrain = map_groups(ytrain)
    mat_ytrain = lb.transform(mapped_ytrain)

    xtest, ytest = get_merged_data(test.data)
    mapped_ytest = map_groups(ytest)
    mat_ytest = lb.transform(mapped_ytest)

    model = create_ete_model((xtrain[0].shape[1:], xtrain[1].shape[1:]), mat_ytrain.shape[1])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

    # train_seq = FCSSequence(train, channels, binarizer=lb, batch_size=128)
    # model.fit_generator(train_seq, epochs=100, use_multiprocessing=True, workers=16)
    class_weight={
        0: 1.0,  # CLL
        1: 2.0,  # MBL
        2: 2.0,  # MCL
        3: 2.0,  # PL
        4: 2.0,  # LPL
        5: 2.0,  # MZL
        6: 10.0,  # FL
        7: 10.0,  # HCL
        8: 1.0,  # normal
    }
    class_weight=None
    model.fit(
        xtrain, mat_ytrain, batch_size=32, epochs=1000, class_weight=class_weight,
        validation_data=(xtest, mat_ytest))

    pred_mat = model.predict(xtest, batch_size=10)
    pred = lb.inverse_transform(pred_mat)
    # print(model.evaluate(xtest, mat_ytest, batch_size=10))
    confusion = sk.metrics.confusion_matrix(mapped_ytest, pred, mapped_groups,)
    print(confusion)

if __name__ == "__main__":
    main()
