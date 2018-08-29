"""
Additional keras objects.
"""
import itertools
import numpy as np

from keras import backend as K
from keras.backend import epsilon
from keras.models import Model
from keras.layers import Input, Dense, Activation


import tensorflow as tf


class WeightedCategoricalCrossEntropy(object):
    """Categorical cross entropy with mask applied to calculated losses."""

    def __init__(self, weights):
        """
        :param weights: list with tuple of ((x, y), weight).
        """
        self.weights = weights
        self.__name__ = 'w_categorical_crossentropy'

    def __call__(self, y_true, y_pred):
        return self.tf_w_categorical_crossentropy(y_true, y_pred)

    def w_categorical_crossentropy(self, y_true, y_pred):
        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        # get the chosen prediction
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        # boolean where prediction equals maximum
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_t, c_p in itertools.product(range(nb_cl), range(nb_cl)):
            w = K.cast(self.weights[c_t, c_p], K.floatx())
            y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
            y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
            final_mask += w * y_p * y_t

        return K.categorical_crossentropy(y_true, y_pred) * final_mask

        # loss = K.categorical_crossentropy(y_true, y_pred)
        # print(loss)
        # return loss

    def tf_w_categorical_crossentropy(self, y_true, y_pred):
        weights = tf.convert_to_tensor(self.weights, tf.float32)

        output_dimensions = list(range(len(y_pred.get_shape())))

        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, -1, True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        loss = - tf.reduce_sum(y_true * tf.log(y_pred), -1)

        pred_binary = tf.cast(tf.equal(
            y_pred,
            tf.expand_dims(tf.reduce_max(y_pred, axis=-1), axis=-1)
        ), tf.float32)
        weight_val = tf.transpose(tf.matmul(
            tf.matmul(y_true, weights), tf.transpose(pred_binary)
        ))
        return loss * weight_val
