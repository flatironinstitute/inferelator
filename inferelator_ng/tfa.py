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

    def tfa(self, dup_self = True):
    	# tf_w_int (tf with interaction): identify TFs that have evidence of TF-gene interactions in prior matrix
        
        # Finds targets that have non-zero regulation
        # TODO: Remove as some form of pre-processing???
        tf_w_int = self.prior.abs().sum(axis = 0) > 0
        tf_w_int = tf_w_int[tf_w_int].index.tolist()

        # subset of prior matrix columns (TFs) that have known TF-gene interactions (tf_w_int == True)
        prior_tf_w_int = self.prior[tf_w_int]

        # dup_tfs: duplicated TFs
        dup_tfs = []
        if dup_self:


        # Everything up til now is useless if the prior is well-made.
        # could replace with checks: check the TF list is            
            duplicates = prior_tf_w_int.transpose().duplicated(keep=False) # mark duplicates as true
            dup_tfs = duplicates[duplicates].index.tolist()

        # find non-duplicated TFs that are also present in target gene list 
        ndup_tfs = list(set(self.prior.columns.values.tolist()).difference(dup_tfs))

        # remove self interactions
        self_tfs = list(set(ndup_tfs).intersection(self.prior.index.values.tolist()))
        self.prior.loc[self_tfs, self_tfs] = 0

        activity = pd.DataFrame(0, index = self.prior.columns, columns = self.exp_mat_halftau.columns)

        if any(tf_w_int):
            activity.loc[tf_w_int,:] = np.matrix(linalg.pinv2(prior_tf_w_int)) * np.matrix(self.exp_mat_halftau)

        tf_wo_int = list(set(self.prior.columns.values.tolist()).difference(tf_w_int))
        activity.loc[tf_wo_int,:] = self.exp_mat.loc[tf_wo_int,:]

        return activity

            




