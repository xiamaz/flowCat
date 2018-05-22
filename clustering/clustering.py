import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tfsom.tfsom import SelfOrganizingMap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import logging
import fcsparser

from compile_cases import CaseCollection


logging.basicConfig(level=logging.WARNING)


class FCSLogTransform(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X, *_):
        names = [n for n in X.columns if "LIN" not in n]
        X[names] = np.log1p(X[names])
        return X

    def fit(self, *_):
        return self


class ClusteringTransform(BaseEstimator, TransformerMixin):
    def __init__(self, m, n, batch_size):
        self.m = m
        self.n = n
        self.batch_size = batch_size

    def fit(self, X, *_):
        # Build the TensorFlow dataset pipeline per the standard tutorial.
        self.graph = tf.Graph()
        with self.graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices(
                X.astype(np.float32)
            )
            num_inputs, dims = X.shape

            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            self.session = tf.Session(
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False
                )
            )
            self._model = SelfOrganizingMap(
                m=self.m, n=self.n, dim=dims, max_epochs=10, gpus=1,
                session=self.session, graph=self.graph, input_tensor=next_element,
                batch_size=self.batch_size, initial_learning_rate=0.05
            )
            init_op = tf.global_variables_initializer()
            self.session.run([init_op])
            self._model.train(num_inputs=num_inputs)

    def transform(self, X, *_):
        result = self._model.transform(X)
        return result / np.sum(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    cases = CaseCollection(sys.argv[1], sys.argv[2])

    batch_size = 2048
    pipe = Pipeline(
        steps=[
            ("log", FCSLogTransform()),
            ("scale", StandardScaler()),
            ("clust", ClusteringTransform(10, 10, batch_size)),
        ]
    )

    for tube in cases.tubes:
        data = cases.get_train_data(num=5, tube=tube)
        pipe.fit(data)

        results = []
        labels = []
        groups = []
        for label, group, testdata in cases.get_all_data(num=300, tube=tube):
            print("Upsampling {}".format(label))
            results.append(pipe.transform(testdata))
            labels.append(label)
            groups.append(group)
        df_all = pd.DataFrame(np.matrix(results))
        df_all["label"] = labels
        df_all["group"] = groups
        outpath = "tube{}.csv".format(tube)
        df_all.to_csv(outpath, sep=";")
