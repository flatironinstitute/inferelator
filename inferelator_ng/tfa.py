import numpy as np
import pandas as pd
from scipy import linalg

class TFA:

    """
    TFA calculates transcription factor activity using matrix pseudoinverse

        Parameters
    --------
    prior: pd.dataframe
        binary or numeric g by t matrix stating existence of gene-TF interactions. 
        g--gene, t--TF.

    exp.mat: pd.dataframe
        normalized expression g by c matrix. g--gene, c--conditions

    exp.mat.halftau: pd.dataframe
        normalized expression matrix for time series.

    dup_self=True: boolean
        If dup_slef (duplicate self) is True, TFs that other TFs with the exact same 
        set of interactions in the prior are kept and will have the same activities
    """

    def __init__(self, prior, exp_mat, exp_mat_halftau):
        self.prior = prior
        self.exp_mat = exp_mat
        self.exp_mat_halftau = exp_mat_halftau

    def tfa(self, allow_self_interactions_for_duplicate_prior_columns = True):
        import pdb; pdb.set_trace()
        # Create activity dataframe with default values set to the expression
    	activity = pd.DataFrame(self.exp_mat.loc[self.prior.columns,:].values, index = self.prior.columns, columns = self.exp_mat.columns)
        
        # Finds tfs that have non-zero regulation
        # TODO: Remove as some form of pre-processing???
        non_zero_tfs = self.prior.loc[:, (self.prior != 0).any(axis=0)].columns.values.tolist()

        # dup_tfs: duplicated TFs
        dup_tfs = []
        if allow_self_interactions_for_duplicate_prior_columns:

        # Everything up til now is useless if the prior is well-made.
        # could replace with checks: check the TF list is            
            duplicates = self.prior[non_zero_tfs].transpose().duplicated(keep=False) # mark duplicates as true
            dup_tfs = duplicates[duplicates].index.tolist()

        # find non-duplicated TFs that are also present in target gene list 
        ndup_tfs = list(set(non_zero_tfs).difference(dup_tfs))
        self_tfs = list(set(ndup_tfs).intersection(self.prior.index.values.tolist()))

        # Set the diagonal of the self-interaction tfs to zero
        subset = self.prior.loc[self_tfs, self_tfs].values
        np.fill_diagonal(subset, 0)
        self.prior.set_value(self_tfs, self_tfs, subset)

        if non_zero_tfs:
            activity.loc[non_zero_tfs,:] = np.matrix(linalg.pinv2(self.prior[non_zero_tfs])) * np.matrix(self.exp_mat_halftau)

        return activity

            




