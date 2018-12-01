"""
Workflow class that splits the prior into a gold standard and new prior
"""

import pandas as pd
import numpy as np
from inferelator_ng import workflow
from inferelator_ng import utils

DEFAULT_SPLIT = 0.5
DEFAULT_AXIS = 0


class PriorGoldStandardSplitWorkflowBase(object):
    prior_gold_split_ratio = DEFAULT_SPLIT
    prior_gold_split_axis = DEFAULT_AXIS

    def set_gold_standard_and_priors(self):
        """
        Ignore the gold standard file. Instead, create a gold standard
        from a 50/50 split of the prior. Half of original prior becomes the new prior,
        the other half becomes the gold standard

        :param gold_standard_split: float
            The proportion of the priors that should go into the gold standard
        :param axis: int
            Splits on rows (when 0), columns (when 1), or on flattened individual data points (when None)
        :return:
            Resets the self.prior_data and self.gold_standard class variables
        """

        # Get the priors
        all_priors = self.input_dataframe(self.priors_file) if self.priors_data is None else self.priors_data.copy()

        # Make sure that the class variables are valid
        assert 1 >= self.prior_gold_split_ratio >= 0, "prior_gold_split_ratio is a ratio between 0 and 1"
        assert self.prior_gold_split_axis in [0, 1, None], "prior_gold_split_axis takes either 0, 1 or None"

        # Split the priors into gold standard based on axis (flatten if axis=None)
        if self.prior_gold_split_axis is None:
            self._split_flattened(all_priors)
        else:
            self._split_axis(all_priors, axis=self.prior_gold_split_axis)

        utils.Debug.vprint("Prior split into a prior {pr} and a gold standard {gs}".format(pr=self.priors_data.shape,
                                                                                           gs=self.gold_standard.shape),
                           level=0)

    def _split_flattened(self, priors):
        pc = np.sum(priors.values != 0)
        gs_count = int(self.prior_gold_split_ratio * pc)
        idx = self._make_shuffled_index(pc)

        pr_idx = priors.values[priors.values != 0].copy()
        gs_idx = priors.values[priors.values != 0].copy()

        pr_idx[idx[0:gs_count]] = 0
        gs_idx[idx[gs_count:]] = 0

        gs = priors.values.copy()
        pr = priors.values.copy()

        gs[gs != 0] = gs_idx
        pr[pr != 0] = pr_idx

        self.priors_data = pd.DataFrame(pr, index=priors.index, columns=priors.columns)
        self.gold_standard = pd.DataFrame(gs, index=priors.index, columns=priors.columns)

    def _split_axis(self, priors, axis=0):
        pc = priors.shape[axis]
        gs_count = int(self.prior_gold_split_ratio * pc)
        idx = self._make_shuffled_index(pc)

        if axis == 0:
            axis_idx = priors.index
        elif axis == 1:
            axis_idx = priors.columns
        else:
            raise ValueError("Axis can only be 0 or 1")

        pr_idx = axis_idx[idx[0:gs_count]]
        gs_idx = axis_idx[idx[gs_count:]]

        self.priors_data = priors.drop(gs_idx, axis=axis)
        self.gold_standard = priors.drop(pr_idx, axis=axis)

    def _make_shuffled_index(self, idx_len):
        idx = list(range(idx_len))
        np.random.shuffle(idx)
        return idx
