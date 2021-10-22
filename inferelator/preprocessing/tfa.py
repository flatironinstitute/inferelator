import numpy as np
import pandas as pd
from scipy import linalg, sparse
from inferelator import utils


class TFA:
    """ TFA calculates transcription factor activity using matrix pseudoinverse """

    def compute_transcription_factor_activity(self, prior, expression_data, expression_data_halftau=None,
                                              keep_self=False):
        """
        Calculate TFA from a prior and expression data object

        :param prior: pd.DataFrame [G x K]
        :param expression_data: InferelatorData [N x G]
        :param expression_data_halftau: InferelatorData [N x G]
        :param keep_self: bool
        :return: InferelatorData [N x K]
        """
        trim_prior, activity_tfs, expr_tfs = self._check_prior(prior, expression_data, keep_self=keep_self)

        activity = np.zeros((expression_data.shape[0], trim_prior.shape[1]), dtype=np.float64)

        if len(activity_tfs) > 0:
            a_cols = trim_prior.columns.isin(activity_tfs)
            expr = expression_data_halftau if expression_data_halftau is not None else expression_data
            activity[:, a_cols] = self._calculate_activity(trim_prior.loc[:, activity_tfs].values, expr)

        if len(expr_tfs) > 0:
            activity[:, trim_prior.columns.isin(expr_tfs)] = expression_data.get_gene_data(expr_tfs, force_dense=True)

        data_name = "Activity" if expression_data.name is None else "{n} Activity".format(n=expression_data.name)
        acti = utils.InferelatorData(activity,
                                     gene_names=trim_prior.columns,
                                     sample_names=expression_data.sample_names,
                                     meta_data=expression_data.meta_data,
                                     name=data_name)
        
        acti.prior_data = prior.copy()
        acti.tfa_prior_data = trim_prior.loc[:, activity_tfs].copy()

        return acti


    def _check_prior(self, prior, expression_data, keep_self=False):
        if not keep_self:
            prior = utils.df_set_diag(prior, 0)

        activity_tfs, expr_tfs, drop_tfs = self._determine_tf_status(prior, expression_data)

        if len(drop_tfs) > 0:
            msg = "{n} TFs are removed from activity (no expression or prior exists)".format(n=len(drop_tfs))
            utils.Debug.vprint(msg, level=0)
            utils.Debug.vprint(" ".join(drop_tfs), level=1)

        prior = prior.drop(drop_tfs, axis=1)

        return prior, activity_tfs, expr_tfs

    @staticmethod
    def _determine_tf_status(prior, expression_data):

        # These TFs can have activity calculations performed for them because there are priors
        activity_tfs = ((prior != 0).sum(axis=0) != 0).values
        # These TFs match gene expression only (no activity calculation)
        expr_tfs = prior.columns.isin(expression_data.gene_names)
        expr_tfs &= ~activity_tfs

        return prior.columns[activity_tfs], prior.columns[expr_tfs], prior.columns[~(activity_tfs | expr_tfs)]

    @staticmethod
    def _calculate_activity(prior, expression_data):

        prior_dtype = np.float32 if expression_data.values.dtype == np.float32 else np.float64
        return TFA._dot_expr_pinv(expression_data.values, sparse.csr_matrix(linalg.pinv(prior).T, dtype=prior_dtype))

    @staticmethod
    def _dot_expr_pinv(expr, inv_prior):
        return utils.DotProduct.dot(expr, inv_prior, dense=True, cast=True)



class NoTFA(TFA):
    """ NoTFA creates an activity matrix from the expression data only """

    def compute_transcription_factor_activity(self, prior, expression_data, expression_data_halftau=None, keep_self=False):
        utils.Debug.vprint("Setting Activity to Expression Values", level=1)
        tf_gene_overlap = prior.columns[prior.columns.isin(expression_data.gene_names)]

        data_name = "Activity" if expression_data.name is None else "{n} Activity".format(n=expression_data.name)
        acti = utils.InferelatorData(expression_data.get_gene_data(tf_gene_overlap, copy=True, force_dense=True),
                                     sample_names=expression_data.sample_names,
                                     meta_data=expression_data.meta_data,
                                     gene_names=tf_gene_overlap,
                                     name=data_name)

        acti.prior_data = prior
        acti.tfa_prior_data = prior.loc[:, tf_gene_overlap].copy()

        return acti

def remove_gene_from_activity(activity_data, gene_expression_data, gene_name, tfs=None):
    """
    Rebuilds activity without the influence of a specific gene

    :param activity_data: Calculated activity matrix (from full prior) [N x K]
    :type activity_data: InferelatorData
    :param gene_expression_data: Gene expression vector for gene i [N, ]
    :type gene_expression_data: np.ndarray
    :param prior_data: Gene name to remove
    :type prior_data: str
    :returns: Calculated activity without the influence of the gene. (Same as if gene X has no expression)
    :rtype: np.ndarray
    """

    n, k = activity_data.shape

    if gene_expression_data.size != n:
        _msg = "Gene expression data expected size {n}; got size {m}".format(n=n, m=gene_expression_data.size)
        raise ValueError(_msg)

    _gene_is_tf = gene_name in activity_data.gene_names

    if activity_data.tfa_prior_data is None:
        _msg = "Prior data is missing from activity matrix {an}".format(an=activity_data.name)
        raise RuntimeError(_msg)

    # If the gene didn't influence the activity calculations, just return the current activities
    if gene_name not in activity_data.tfa_prior_data.index:
        return activity_data.values

    # Check and see if the gene is a TF; if it is, check to see if it's expression was used as a proxy for activity
    if _gene_is_tf and (gene_name not in activity_data.tfa_prior_data.columns or
                        all(activity_data.tfa_prior_data.loc[:, gene_name] == 0)):
        _gene_tf_idx = activity_data.gene_names.get_loc(gene_name)
        if np.allclose(activity_data.values[:, _gene_tf_idx], gene_expression_data.flatten()):
            activity_data = activity_data.values.copy()
            activity_data[:, _gene_tf_idx] = 0
            return activity_data

    # Check for no prior
    if activity_data.tfa_prior_data.size == 0:
        return activity_data.values
    elif all(activity_data.tfa_prior_data.loc[gene_name, :] == 0):
        return activity_data.values

    # Calculate the prior pseudoinverse
    gene_prior_effect = pd.DataFrame(linalg.pinv(activity_data.tfa_prior_data.values).T,
                                     index=activity_data.tfa_prior_data.index,
                                     columns=activity_data.tfa_prior_data.columns)

    # Stretch it with zeros to match the width of the activity matrix and select the gene row
    gene_prior_effect = gene_prior_effect.reindex(activity_data.gene_names, axis=1).fillna(0)
    gene_prior_effect = gene_prior_effect.loc[gene_name, :].values.reshape(1, k)

    # Calculate the effect of the gene on the prior activity
    # Then remove it
    fixed_activity = TFA._dot_expr_pinv(gene_expression_data.reshape(n, 1), gene_prior_effect)
    fixed_activity *= -1
    fixed_activity += activity_data.values

    # Subset down to a set of TFs if passed
    if tfs is not None:
        fixed_activity = fixed_activity[:, activity_data.gene_names.get_indexer(tfs)]

    return fixed_activity
