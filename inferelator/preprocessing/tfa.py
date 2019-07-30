import numpy as np
import pandas as pd
from scipy import linalg

from inferelator import utils
from inferelator.utils import Validator as check


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

        assert check.dataframes_align([expression_matrix, expression_matrix_halftau])
        assert check.indexes_align((prior.index, expression_matrix.index), check_order=False)

        self.prior = prior
        self.expression_matrix = expression_matrix
        self.expression_matrix_halftau = expression_matrix_halftau

    def compute_transcription_factor_activity(self, allow_self_interactions_for_duplicate_prior_columns=True):

        activity, self.prior, non_zero_tfs = process_expression_into_activity(self.expression_matrix, self.prior)
        self.fix_self_interacting(non_zero_tfs, allow_duplicates=allow_self_interactions_for_duplicate_prior_columns)

        # Set the activity of non-zero tfs to the pseudoinverse of the prior matrix times the expression
        if len(non_zero_tfs) > 0:
            activity.loc[non_zero_tfs, :] = np.matrix(linalg.pinv2(self.prior[non_zero_tfs])) * np.matrix(
                self.expression_matrix_halftau)
        else:
            utils.Debug.vprint("No prior information for TFs exists. Using expression for TFA exclusively.", level=0)

        activity_nas = activity.isna().any(axis=0)
        if activity_nas.sum() > 0:
            lose_tfs = activity_nas.index[activity_nas].tolist()
            utils.Debug.vprint("Dropping TFs with NaN values: {drop}".format(drop=" ".join(lose_tfs)))
            activity = activity.dropna(axis=0)

        return activity

    def fix_self_interacting(self, non_zero_tfs, allow_duplicates=True):
        # Find all non-zero TFs that are duplicates of any other non-zero tfs
        is_duplicated = self.prior[non_zero_tfs].transpose().duplicated(keep=False)

        # Find non-zero TFs that are also present in target gene list
        self_interacting_tfs = non_zero_tfs.intersection(self.prior.index)

        if is_duplicated.sum() > 0:
            duplicates = is_duplicated[is_duplicated].index.tolist()

            # If this flag is set to true, don't count duplicates as self-interacting when setting the diag to zero
            if allow_duplicates:
                self_interacting_tfs = self_interacting_tfs.difference(duplicates)

        # Set the diagonal of the matrix subset of self-interacting tfs to zero
        subset = self.prior.loc[self_interacting_tfs, self_interacting_tfs].values
        np.fill_diagonal(subset, 0)
        self.prior.at[self_interacting_tfs, self_interacting_tfs] = subset


class NoTFA(TFA):

    def compute_transcription_factor_activity(self, allow_self_interactions_for_duplicate_prior_columns=True):
        utils.Debug.vprint("Setting Activity to Expression Values", level=1)

        # Get the activity matrix with expression data only
        activity, _, _ = process_expression_into_activity(self.expression_matrix, self.prior)

        # Return only TFs which we have expression data for
        activity = activity.loc[activity.index.intersection(self.expression_matrix.index), :]
        return activity


def process_expression_into_activity(expression_matrix, prior):
    """
    Create a [K x N] activity matrix which is populated with the expression values for each TF. Remove any TF which
    has no prior information and no expression data.

    :param expression_matrix: pd.DataFrame [G x N]
    :param prior: pd.DataFrame [G x K]
    :return activity, prior, non_zero_tfs: pd.DataFrame [k x N], pd.DataFrame [G x k], pd.Index [k]
    """
    # Find TFs that have non-zero columns in the priors matrix
    non_zero_tfs = pd.Index(prior.columns[(prior != 0).any(axis=0)])
    # Delete tfs that have neither prior information nor expression
    delete_tfs = prior.columns.difference(expression_matrix.index).difference(non_zero_tfs)

    # Raise warnings
    if len(delete_tfs) > 0:
        message = "{num} TFs are removed from activity (no expression or prior exists)".format(num=len(delete_tfs))
        utils.Debug.vprint(message, level=0)
        prior = prior.drop(delete_tfs, axis=1)

    # Create activity dataframe with values set by default to the transcription factor's expression
    # Create a dataframe [K x N] with the expression values as defaults
    activity = expression_matrix.reindex(prior.columns)

    return activity, prior, non_zero_tfs
