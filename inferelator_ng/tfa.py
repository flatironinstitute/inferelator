import numpy as np
import pandas as pd
from scipy import linalg
import warnings

class TFA:

    """
    TFA calculates transcription factor activity using matrix pseudoinverse

        Parameters
    --------
    prior: pd.dataframe
        binary or numeric g by t matrix stating existence of gene-TF interactions. 
        g: gene, t: TF.

    expression_matrix: pd.dataframe
        normalized expression g by c matrix. g--gene, c--conditions

    expression_matrix_halftau: pd.dataframe
        normalized expression matrix for time series.

    allow_self_interactions_for_duplicate_prior_columns=True: boolean
        If True, TFs that are identical to other columns in the prior matrix 
        do not have their self-interactios removed from the prior
        and therefore will have the same activities as their duplicate tfs.
    """

    def __init__(self, prior, expression_matrix, expression_matrix_halftau):
        self.prior = prior
        self.expression_matrix = expression_matrix
        self.expression_matrix_halftau = expression_matrix_halftau

    def compute_transcription_factor_activity(self, allow_self_interactions_for_duplicate_prior_columns = True):
        # Find TFs that have non-zero columns in the priors matrix
        non_zero_tfs = self.prior.columns[(self.prior != 0).any(axis=0)].tolist()

        # Delete tfs that have neither prior information nor expression
        delete_tfs = set(self.prior.columns).difference(self.prior.index).difference(non_zero_tfs)
        # Raise warnings
        if len(delete_tfs) > 0:
            message = " ".join([str(len(delete_tfs)).capitalize(),
             "transcription factors are removed because no expression or prior information exists."])
            warnings.warn(message)
            self.prior = self.prior.drop(delete_tfs, axis = 1)

        # Create activity dataframe with values set by default to the transcription factor's expression
        activity = pd.DataFrame(self.expression_matrix.loc[self.prior.columns,:].values,
                index = self.prior.columns,
                columns = self.expression_matrix.columns)

        # Find all non-zero TFs that are duplicates of any other non-zero tfs
        is_duplicated = self.prior[non_zero_tfs].transpose().duplicated(keep=False)
        duplicates = is_duplicated[is_duplicated].index.tolist()

        # Find non-zero TFs that are also present in target gene list 
        self_interacting_tfs = set(non_zero_tfs).intersection(self.prior.index)

        # If this flag is set to true, don't count duplicates as self-interacting when setting the diag to zero
        if allow_self_interactions_for_duplicate_prior_columns:
            self_interacting_tfs = self_interacting_tfs.difference(duplicates)

        # Set the diagonal of the matrix subset of self-interacting tfs to zero
        subset = self.prior.loc[self_interacting_tfs, self_interacting_tfs].values
        np.fill_diagonal(subset, 0)
        self.prior.at[self_interacting_tfs, self_interacting_tfs] = subset

        # Set the activity of non-zero tfs to the pseudoinverse of the prior matrix times the expression
        if non_zero_tfs:
            activity.loc[non_zero_tfs,:] = np.matrix(linalg.pinv2(self.prior[non_zero_tfs])) * np.matrix(self.expression_matrix_halftau)

        return activity

            




