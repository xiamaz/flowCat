import numpy as np
from numpy.testing import assert_array_equal
import tensorflow as tf
from flowcat.utils import classification_utils
from . import shared


class ClassificationTestCase(shared.FlowcatTestCase):

    def test_group_weights(self):
        cases = [
            (
                {"a": 1},
                {"a": 1.0},
            ),
            (
                {"a": 10, "b": 10},
                {"a": 1.0, "b": 1.0},
            ),
            (
                {"a": 10, "b": 10, "c": 10},
                {"a": 1.0, "b": 1.0, "c": 1.0},
            ),
            (
                {"a": 1, "b": 1, "c": 1},
                {"a": 1.0, "b": 1.0, "c": 1.0},
            ),
            (
                {"a": 100, "b": 10},
                {"a": 1.0, "b": 10.0},
            ),
            (
                {"a": 100, "b": 1000},
                {"a": 10.0, "b": 1.0},
            ),
            (
                {"a": 3, "b": 1, "c": 1},
                {"a": 1.0, "b": 3.0, "c": 3.0},
            ),
        ]
        for data, expected in cases:
            result = classification_utils.calculate_group_weights(data)
            self.assertEqual(result, expected)

    def test_categorical_crossentropy(self):
        cost_mat = np.array([
            [1, 2, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])
        wcc = classification_utils.WeightedCategoricalCrossentropy(cost_mat)
        y_true_mat = np.array([[1, 0, 0]])
        y_pred_mat = np.array([[0, 1, 0]])
        with tf.Session() as sess:
            y_true = tf.constant(y_true_mat, dtype=tf.float32)
            y_pred = tf.constant(y_pred_mat, dtype=tf.float32)
            res = wcc(y_true, y_pred)
            exp = classification_utils.categorical_crossentropy(y_true, y_pred)
            res_val, exp_val = sess.run([res, exp])
            self.assertEqual(res_val, exp_val * 2)

    def test_cost_mapping(self):
        cases = [
            (
                (
                    {("a", "b"): 2, },
                    ("a", "b", "c")
                ),
                np.array([
                    [1, 2, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ], np.float32)
            ),
            (
                (
                    {("a", "b"): 2, ("a", "a"): 3},
                    ("a", "b", "c")
                ),
                np.array([
                    [3, 2, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ], np.float32)
            ),
        ]

        for args, expected in cases:
            result = classification_utils.build_cost_matrix(*args)
            assert_array_equal(result, expected)
