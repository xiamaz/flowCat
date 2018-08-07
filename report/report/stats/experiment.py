"""Methods for statistical analysis of classfication experiments.
"""
# independent ttest
import numpy as np
from scipy.stats import ttest_ind


def ttest_exp_sets(baseset, testset, testval=("t1", "micro")):
    """Take two experiment sets and return the p-value and individual means
    in the tested value.
    We assume the two experiment sets to be independent of each other.
    :param baseset: Dataframe containing experiment information.
    :param testset: Another experiment info dataframe to test against.
    :param testval: function returning a list of values for each set.

    :return: tuple with mean a, mean b, p_value
    """

    basevals = _select_method(baseset, *testval)
    testvals = _select_method(testset, *testval)

    basemean = np.mean(basevals)
    testmean = np.mean(testvals)

    tstat, pvalue = ttest_ind(basevals, testvals, equal_var=False)
    return {
        "mean_a": basemean,
        "n_a": len(basevals),
        "mean_b": testmean,
        "n_b": len(testvals),
        "p_value": pvalue,
    }


def _select_method(statset, method, avg_type):
    """Get the method column in the data."""
    selection = statset.loc[
        statset.index.get_level_values("method") == method,
        avg_type
    ]
    return selection
