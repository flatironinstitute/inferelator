"""
Workflow class that splits the prior into a gold standard and new prior
"""

import random
import pandas as pd
import numpy as np
from . import workflow

DEFAULT_SPLIT = 0.5


class PriorGoldStandardSplitWorkflowBase(object):
    prior_gold_split = None

    def set_gold_standard_and_priors(self, gold_standard_split=DEFAULT_SPLIT, axis=0):
        """
        Ignore the gold standard file. Instead, create a gold standard
        from a 50/50 split of the prior. Half of original prior becomes the new prior, 
        the other half becomes the gold standard
        """

        # Get the priors
        all_priors = self.input_dataframe(self.priors_file) if self.priors_data is None else self.priors_data.copy()

        # Set the split ratio
        self.prior_gold_split = self.prior_gold_split if self.prior_gold_split is not None else gold_standard_split

        # Split the priors into gold standard based on axis (flatten if axis=None)
        self._split_flattened(all_priors) if axis is None else self._split_axis(all_priors, axis=axis)

    def _split_flattened(self, priors):
        pc = np.sum(priors.values != 0)
        gs_count = int(self.prior_gold_split * pc)
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
        gs_count = int(self.prior_gold_split * pc)
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
