"""
Workflow class that splits the prior into a gold standard and new prior
"""

import pandas as pd
import numpy as np
from inferelator_ng import utils

DEFAULT_SPLIT = 0.5
DEFAULT_AXIS = 0


def split_priors_for_gold_standard(all_priors, split_ratio=DEFAULT_SPLIT, split_axis=DEFAULT_AXIS):
    """
    Ignore the gold standard file. Instead, create a gold standard
    from a 50/50 split of the prior. Half of original prior becomes the new prior,
    the other half becomes the gold standard

    :param all_priors: pd.DataFrame [G x K]
        Prior data
    :param gold_standard_split: float
        The proportion of the priors that should go into the gold standard
    :param axis: int
        Splits on rows (when 0), columns (when 1), or on flattened individual data points (when None)
    :return prior_data, gold_standard: pd.DataFrame [G/2 x K], pd.DataFrame [G/2 x K]
        Returns a new prior and gold standard by splitting the old one in half
    """

    # Get the priors

    # Make sure that the class variables are valid
    if not 1 >= split_ratio >= 0:
        raise ValueError("prior_gold_split_ratio is a ratio between 0 and 1")

    if not (split_axis in [0, 1, None]):
        raise ValueError("prior_gold_split_axis takes either 0, 1 or None")

    # Split the priors into gold standard based on axis (flatten if axis=None)
    if split_axis is None:
        priors_data, gold_standard = _split_flattened(all_priors)
    else:
        priors_data, gold_standard = _split_axis(all_priors, axis=split_axis)

    utils.Debug.vprint("Prior split into a prior {pr} and a gold standard {gs}".format(pr=priors_data.shape,
                                                                                       gs=gold_standard.shape), level=0)

    return priors_data, gold_standard


def _split_flattened(priors, split_ratio=DEFAULT_SPLIT):
    pc = np.sum(priors.values != 0)
    gs_count = int(split_ratio * pc)
    idx = _make_shuffled_index(pc)

    pr_idx = priors.values[priors.values != 0].copy()
    gs_idx = priors.values[priors.values != 0].copy()

    pr_idx[idx[0:gs_count]] = 0
    gs_idx[idx[gs_count:]] = 0

    gs = priors.values.copy()
    pr = priors.values.copy()

    gs[gs != 0] = gs_idx
    pr[pr != 0] = pr_idx

    priors_data = pd.DataFrame(pr, index=priors.index, columns=priors.columns)
    gold_standard = pd.DataFrame(gs, index=priors.index, columns=priors.columns)

    return priors_data, gold_standard


def _split_axis(priors, axis=DEFAULT_AXIS, split_ratio=DEFAULT_SPLIT):
    pc = priors.shape[axis]
    gs_count = int(split_ratio * pc)
    idx = _make_shuffled_index(pc)

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


def _make_shuffled_index(idx_len):
    idx = list(range(idx_len))
    np.random.shuffle(idx)
    return idx
