import pandas as pd
import numpy as np

from inferelator_ng import utils
from inferelator_ng import bayes_stats
from inferelator_ng import regression
from inferelator_ng import mi
from inferelator_ng.distributed.kvs_controller import KVSController

# Default number of predictors to include in the model
DEFAULT_nS = 10

# Default weight for priors & Non-priors
# If this is the same as no_prior_weight:
#   Priors will be included in the pp matrix before the number of predictors is reduced to nS
#   They won't get special treatment in the model though
DEFAULT_prior_weight = 1
DEFAULT_no_prior_weight = 1

# Throw away the priors which have a CLR that is 0 before the number of predictors is reduced by BIC
DEFAULT_filter_priors_for_clr = False


class BBSR(regression.BaseRegression):
    # Bayseian correlation measurements
    clr_mat = None  # [G x K] float

    # Priors Data
    prior_mat = None  # [G x K] # numeric
    filter_priors_for_clr = DEFAULT_filter_priors_for_clr  # bool

    # Weights for Predictors (weights_mat is set with _calc_weight_matrix)
    weights_mat = None  # [G x K] numeric
    prior_weight = DEFAULT_prior_weight  # numeric
    no_prior_weight = DEFAULT_no_prior_weight  # numeric

    # Predictors to include in modeling (pp is set with _build_pp_matrix)
    pp = None  # [G x K] bool
    nS = DEFAULT_nS  # int

    def __init__(self, X, Y, clr_mat, prior_mat, nS=DEFAULT_nS, prior_weight=DEFAULT_prior_weight,
                 no_prior_weight=DEFAULT_no_prior_weight, chunk=regression.DEFAULT_CHUNK):
        """
        Create a Regression object for Bayes Best Subset Regression

        :param X: pd.DataFrame [K x N]
            Expression / Activity data
        :param Y: pd.DataFrame [G x N]
            Response data
        :param clr_mat: pd.DataFrame [G x K]
            Calculated CLR between features of X & Y
        :param prior_mat: pd.DataFrame [G x K]
            Prior data between features of X & Y
        :param nS: int
            Number of predictors to retain
        :param prior_weight: int
            Weight of a predictor which does have a prior
        :param no_prior_weight: int
            Weight of a predictor which doesn't have a prior
        """

        super(BBSR, self).__init__(X, Y, chunk=chunk)

        self.nS = nS

        # Calculate the weight matrix
        self.prior_weight = prior_weight
        self.no_prior_weight = no_prior_weight
        weights_mat = self._calculate_weight_matrix(prior_mat, p_weight=prior_weight, no_p_weight=no_prior_weight)
        utils.Debug.vprint("Weight matrix {} construction complete".format(weights_mat.shape))

        # Rebuild weights, priors, and the CLR matrix for the features that are in this bootstrap
        self.weights_mat = weights_mat.loc[self.genes, self.tfs]
        self.prior_mat = prior_mat.loc[self.genes, self.tfs]
        self.clr_mat = clr_mat.loc[self.genes, self.tfs]

        # Build a boolean matrix indicating which tfs should be used as predictors for regression for each gene
        self.pp = self._build_pp_matrix()

    def run(self):
        """
        Execute BBSR

        :return: pd.DataFrame [G x K], pd.DataFrame [G x K]
            Returns the regression betas and beta error reductions for all threads if this is the master thread (rank 0)
            Returns None, None if it's a subordinate thread
        """

        def regression_maker(regression_obj, j):
            level = 0 if j % 100 == 0 else 2
            utils.Debug.vprint(regression.PROGRESS_STR.format(gn=self.genes[j], i=j, total=self.G), level=level)
            data = bayes_stats.bbsr(regression_obj.X.values,
                                    regression_obj.Y.iloc[j, :].values,
                                    regression_obj.pp.iloc[j, :].values,
                                    regression_obj.weights_mat.iloc[j, :].values,
                                    regression_obj.nS)
            data['ind'] = j
            return data

        dsk = {'j': list(range(self.G)), 'data': (regression_maker, self, 'j')}
        run_data = KVSController.get(dsk, 'data', tell_children=False)

        if KVSController.is_master:
            return self.pileup_data(run_data)
        else:
            return None, None

    def _build_pp_matrix(self):
        """
        From priors and context likelihood of relatedness, determine which predictors should be included in the model

        :return pp: pd.DataFrame [G x K]
            Boolean matrix indicating which predictor variables should be included in BBSR for each response variable
        """

        # Create a predictor boolean array from priors
        pp = np.logical_or(self.prior_mat != 0, self.weights_mat != self.no_prior_weight)

        pp_idx = pp.index
        pp_col = pp.columns

        if self.filter_priors_for_clr:
            # Set priors which have a CLR of 0 to FALSE
            pp = np.logical_and(pp, self.clr_mat != 0).values
        else:
            pp = pp.values

        # Mark the nS predictors with the highest CLR true (Do not include anything with a CLR of 0)
        mask = np.logical_or(self.clr_mat == 0, ~np.isfinite(self.clr_mat)).values
        masked_clr = np.ma.array(self.clr_mat.values, mask=mask)
        for i in range(self.G):
            n_to_keep = min(self.nS, self.K, mask.shape[1] - np.sum(mask[i, :]))
            if n_to_keep == 0:
                continue
            clrs = np.ma.argsort(masked_clr[i, :], endwith=False)[-1 * n_to_keep:]
            pp[i, clrs] = True

        # Rebuild into a DataFrame and set autoregulation to 0
        pp = pd.DataFrame(pp, index=pp_idx, columns=pp_col, dtype=np.dtype(bool))
        pp = utils.df_set_diag(pp, False)

        return pp

    @staticmethod
    def _calculate_weight_matrix(p_matrix, no_p_weight=DEFAULT_no_prior_weight, p_weight=DEFAULT_prior_weight):
        """
        Create a weights matrix. Everywhere p_matrix is not set to 0, the weights matrix will have p_weight. Everywhere
        p_matrix is set to 0, the weights matrix will have no_p_weight
        :param p_matrix: pd.DataFrame [G x K]
        :param no_p_weight: int
            Weight of something which doesn't have a prior
        :param p_weight: int
            Weight of something which does have a prior
        :return weights_mat: pd.DataFrame [G x K]
        """
        weights_mat = p_matrix * 0 + no_p_weight
        return weights_mat.mask(p_matrix != 0, other=p_weight)


def patch_workflow(obj):
    """
    Add BBSR regression into a TFAWorkflow object

    :param obj: TFAWorkflow
    """

    import types

    if not hasattr(obj, 'mi_driver'):
        obj.mi_driver = mi.MIDriver
    if not hasattr(obj, 'mi_sync_path'):
        obj.mi_sync_path = None

    def run_bootstrap(self, bootstrap):
        X = self.design.iloc[:, bootstrap]
        Y = self.response.iloc[:, bootstrap]
        utils.Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=0)
        clr_matrix, mi_matrix = self.mi_driver(sync_in_tmp_path=self.mi_sync_path).run(X, Y)
        mi_matrix = None
        utils.Debug.vprint('Calculating betas using BBSR', level=0)

        return BBSR(X, Y, clr_matrix, self.priors_data).run()

    obj.run_bootstrap = types.MethodType(run_bootstrap, obj)
