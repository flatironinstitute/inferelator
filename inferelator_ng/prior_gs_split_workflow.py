"""
Workflow class that splits the prior into a gold standard and new prior
"""

import random
import pandas as pd
import numpy as np
from . import workflow

class PriorGoldStandardSplitWorkflowBase(workflow.WorkflowBase):

    def set_gold_standard_and_priors(self):
        """
        Ignore the gold standard file. Instead, create a gold standard
        from a 50/50 split of the prior. Half of original prior becomes the new prior, 
        the other half becomes the gold standard
        """
        split_ratio = 0.5
        self.priors_data = self.input_dataframe(self.priors_file)
        prior = pd.melt(self.priors_data.reset_index(), id_vars='index')
        prior_edges = prior.index[prior.value != 0]
        keep = np.random.choice(prior_edges, int(len(prior_edges)*split_ratio), replace=False)
        prior_subsample = prior.copy(deep=True)
        gs_subsample = prior.copy(deep=True)
        prior_subsample.loc[prior_edges[~prior_edges.isin(keep)], 'value'] = 0
        gs_subsample.loc[prior_edges[prior_edges.isin(keep)], 'value'] = 0
        prior_subsample = pd.pivot_table(prior_subsample, index='index', columns='variable', values='value', fill_value=0)
        gs_subsample = pd.pivot_table(gs_subsample, index='index', columns='variable', values='value', fill_value=0)
        self.priors_data = prior_subsample
        self.gold_standard = gs_subsample