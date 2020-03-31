from typing import Dict
import numpy as np
import keras.backend as K
from keras.losses import categorical_crossentropy


def calculate_group_weights(group_count: Dict[str, int]) -> Dict[int, float]:
    """Calculate group weightings based on number of cases present.
    """
    num_cases = sum(group_count.values())
    balanced_nums = num_cases / len(group_count)
    balanced_loss_weights = {g: balanced_nums / group_count[g] for g in group_count}
    min_ratio = min(balanced_loss_weights.values())
    balanced_loss_weights = {g: v / min_ratio for g, v in balanced_loss_weights.items()}
    return balanced_loss_weights


def build_cost_matrix(cost_mapping: Dict[tuple, float], groups: list):
    """Map a dict of tuples from true to predicted class to a cost matrix with rows and cols according to group list."""
    cost_matrix = np.ones((len(groups), len(groups)), np.float32)
    for (group_a, group_b), value in cost_mapping.items():
        ida = groups.index(group_a)
        idb = groups.index(group_b)
        cost_matrix[ida, idb] = value
        # cost_matrix[idb, ida] = value
    return cost_matrix


class WeightedCategoricalCrossentropy:
    """Source: https://github.com/keras-team/keras/issues/2115#issuecomment-530762739"""

    def __init__(self, cost_mat, name='weighted_categorical_crossentropy', **kwargs):
        assert(cost_mat.ndim == 2)
        assert(cost_mat.shape[0] == cost_mat.shape[1])
        self.cost_mat = K.cast_to_floatx(cost_mat)

    def __call__(self, y_true, y_pred, sample_weight=None):
        cost_weight = get_sample_weights(y_true, y_pred, self.cost_mat)
        # cost_weight = K.print_tensor(cost_weight)
        return categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred,
        ) * cost_weight


def get_sample_weights(y_true, y_pred, cost_m):
    num_classes = len(cost_m)

    # y_pred.shape.assert_has_rank(2)
    # y_pred.shape[1].assert_is_compatible_with(num_classes)
    # y_pred.shape.assert_is_compatible_with(y_true.shape)

    y_pred_oh = K.one_hot(K.argmax(y_pred), num_classes)

    y_true_nk1 = K.expand_dims(y_true, 2)
    y_pred_n1k = K.expand_dims(y_pred_oh, 1)
    cost_m_1kk = K.expand_dims(cost_m, 0)

    sample_weights_nkk = cost_m_1kk * y_true_nk1 * y_pred_n1k
    sample_weights_n = K.sum(sample_weights_nkk, axis=[1, 2])

    return sample_weights_n
