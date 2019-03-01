"""
Workflow class that splits the prior into a gold standard and new prior
"""

import pandas as pd
import numpy as np
from inferelator_ng.utils import Validator as check
from inferelator_ng import default


def split_for_cv(all_data, split_ratio, split_axis=default.DEFAULT_CV_AXIS, seed=default.DEFAULT_CV_RANDOM_SEED):
    """
    Take a dataframe and split it according to split_ratio on split_axis into two new dataframes. This is for
    crossvalidation splits of a gold standard

    :param all_data: pd.DataFrame [G x K]
        Existing prior or gold standard data
    :param split_ratio: float
        The proportion of the priors that should go into the gold standard
    :param split_axis: int
        Splits on rows (when 0), columns (when 1), or on flattened individual data points (when None)
    :return prior_data, gold_standard: pd.DataFrame [G/2 x K], pd.DataFrame [G/2 x K]
        Returns a new prior and gold standard by splitting the old one in half
    """

    check.argument_numeric(split_ratio, 0, 1)
    check.argument_enum(split_axis, [0, 1], allow_none=True)

    # Split the priors into gold standard based on axis (flatten if axis=None)
    if split_axis is None:
        priors_data, gold_standard = _split_flattened(all_data, split_ratio, seed=seed)
    else:
        priors_data, gold_standard = _split_axis(all_data, split_ratio, axis=split_axis, seed=seed)

    return priors_data, gold_standard


def remove_prior_circularity(priors, gold_standard, split_axis=default.DEFAULT_CV_AXIS):
    """
    Remove all row labels that occur in the gold standard from the prior
    :param priors: pd.DataFrame [M x N]
    :param gold_standard: pd.DataFrame [m x n]
    :param split_axis: int (0,1)
    :return new_priors: pd.DataFrame [M-m x N]
    :return gold_standard: pd.DataFrame [m x n]
    """

    check.argument_enum(split_axis, [0, 1])
    new_priors = priors.drop(gold_standard.axes[split_axis], axis=split_axis, errors='ignore')

    return new_priors, gold_standard


def _split_flattened(data, split_ratio, seed=default.DEFAULT_CV_RANDOM_SEED):
    """
    Instead of splitting by axis labels, split edges and ignore axes
    :param data: pd.DataFrame [M x N]
    :param split_ratio: float
    :param seed:
    :return priors_data: pd.DataFrame [M x N]
    :return gold_standard: pd.DataFrame [M x N]
    """

    check.argument_numeric(split_ratio, 0, 1)

    pc = np.sum(data.values != 0)
    gs_count = int(split_ratio * pc)
    idx = _make_shuffled_index(pc, seed=seed)

    pr_idx = data.values[data.values != 0].copy()
    gs_idx = data.values[data.values != 0].copy()

    pr_idx[idx[0:gs_count]] = 0
    gs_idx[idx[gs_count:]] = 0

    gs = data.values.copy()
    pr = data.values.copy()

    gs[gs != 0] = gs_idx
    pr[pr != 0] = pr_idx

    priors_data = pd.DataFrame(pr, index=data.index, columns=data.columns)
    gold_standard = pd.DataFrame(gs, index=data.index, columns=data.columns)

    return priors_data, gold_standard


def _split_axis(priors, split_ratio, axis=default.DEFAULT_CV_AXIS, seed=default.DEFAULT_CV_RANDOM_SEED):
    """
    Split by axis labels on the chosen axis
    :param priors: pd.DataFrame [M x N]
    :param split_ratio: float
    :param axis: [0, 1]
    :param seed:
    :return:
    """

    check.argument_numeric(split_ratio, 0, 1)
    check.argument_enum(axis, [0, 1])

    pc = priors.shape[axis]
    gs_count = int((1 - split_ratio) * pc)
    idx = _make_shuffled_index(pc, seed=seed)

    if axis == 0:
        axis_idx = priors.index
    elif axis == 1:
        axis_idx = priors.columns
    else:
        raise ValueError("Axis can only be 0 or 1")

    pr_idx = axis_idx[idx[0:gs_count]]
    gs_idx = axis_idx[idx[gs_count:]]

    priors_data = priors.drop(gs_idx, axis=axis)
    gold_standard = priors.drop(pr_idx, axis=axis)

    return priors_data, gold_standard


def _make_shuffled_index(idx_len, seed=default.DEFAULT_CV_RANDOM_SEED):
    idx = list(range(idx_len))
    np.random.RandomState(seed=seed).shuffle(idx)
    return idx
