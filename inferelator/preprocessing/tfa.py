import numpy as np
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

        prior, activity_tfs, expr_tfs = self._check_prior(prior, expression_data, keep_self=keep_self)

        activity = np.zeros((expression_data.shape[0], prior.shape[1]), dtype=np.float64)

        if len(activity_tfs) > 0:
            a_cols = prior.columns.isin(activity_tfs)
            expr = expression_data_halftau if expression_data_halftau is not None else expression_data
            activity[:, a_cols] = self._calculate_activity(prior.loc[:, activity_tfs].values, expr)

        if len(expr_tfs) > 0:
            activity[:, prior.columns.isin(expr_tfs)] = expression_data.get_gene_data(expr_tfs, force_dense=True)

        data_name = "Activity" if expression_data.name is None else "{n} Activity".format(n=expression_data.name)
        return utils.InferelatorData(activity,
                                     gene_names=prior.columns,
                                     sample_names=expression_data.sample_names,
                                     meta_data=expression_data.meta_data,
                                     name=data_name)

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
        return utils.DotProduct.dot(expression_data.values, sparse.csr_matrix(linalg.pinv2(prior).T, dtype=prior_dtype),
                                    dense=True, cast=True)


class NoTFA(TFA):
    """ NoTFA creates an activity matrix from the expression data only """

    def compute_transcription_factor_activity(self, prior, expression_data, expression_data_halftau=None, keep_self=False):
        utils.Debug.vprint("Setting Activity to Expression Values", level=1)
        tf_gene_overlap = prior.columns[prior.columns.isin(expression_data.gene_names)]

        data_name = "Activity" if expression_data.name is None else "{n} Activity".format(n=expression_data.name)
        return utils.InferelatorData(expression_data.get_gene_data(tf_gene_overlap, copy=True, force_dense=True),
                                     sample_names=expression_data.sample_names,
                                     meta_data=expression_data.meta_data,
                                     gene_names=tf_gene_overlap,
                                     name=data_name)
