import numpy as np
import pandas as pd
from scipy import linalg

from inferelator_ng import utils

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
        non_zero_tfs = pd.Index(self.prior.columns[(self.prior != 0).any(axis=0)])
        # Delete tfs that have neither prior information nor expression
        delete_tfs = self.prior.columns.difference(self.expression_matrix.index).difference(non_zero_tfs)

        # Raise warnings
        if len(delete_tfs) > 0:
            message = "{num} TFs are removed from activity (no expression or prior exists)".format(num=len(delete_tfs))
            utils.Debug.vprint(message, level=0)
            self.prior = self.prior.drop(delete_tfs, axis = 1)

        # Create activity dataframe with values set by default to the transcription factor's expression
        # Create an empty dataframe [K x G]
        activity = pd.DataFrame(0.0, index=self.prior.columns, columns=self.expression_matrix.columns)

        # Populate with expression values as a default
        add_default_activity = self.prior.columns.intersection(self.expression_matrix.index)
        activity.loc[add_default_activity, :] = self.expression_matrix.loc[add_default_activity, :]

        # Find all non-zero TFs that are duplicates of any other non-zero tfs
        is_duplicated = self.prior[non_zero_tfs].transpose().duplicated(keep=False)

        # Find non-zero TFs that are also present in target gene list
        self_interacting_tfs = non_zero_tfs.intersection(self.prior.index)

        if is_duplicated.sum() > 0:
            duplicates = is_duplicated[is_duplicated].index.tolist()

            # If this flag is set to true, don't count duplicates as self-interacting when setting the diag to zero
            if allow_self_interactions_for_duplicate_prior_columns:
                self_interacting_tfs = self_interacting_tfs.difference(duplicates)

        # Set the diagonal of the matrix subset of self-interacting tfs to zero
        subset = self.prior.loc[self_interacting_tfs, self_interacting_tfs].values
        np.fill_diagonal(subset, 0)
        self.prior.at[self_interacting_tfs, self_interacting_tfs] = subset

        # Set the activity of non-zero tfs to the pseudoinverse of the prior matrix times the expression
        if len(non_zero_tfs) > 0:
            activity.loc[non_zero_tfs,:] = np.matrix(linalg.pinv2(self.prior[non_zero_tfs])) * np.matrix(self.expression_matrix_halftau)

        return activity

            




