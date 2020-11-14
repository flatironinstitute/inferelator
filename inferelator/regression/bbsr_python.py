import pandas as pd
import numpy as np

from inferelator import utils
from inferelator.regression import bayes_stats
from inferelator.regression import base_regression
from inferelator.regression import mi
from inferelator.distributed.inferelator_mp import MPControl

# Default number of predictors to include in the model
DEFAULT_nS = 10

# Default weight for priors & Non-priors
# If prior_weight is the same as no_prior_weight:
#   Priors will be included in the pp matrix before the number of predictors is reduced to nS
#   They won't get special treatment in the model though
DEFAULT_prior_weight = 1
DEFAULT_no_prior_weight = 1

# Throw away the priors which have a CLR that is 0 before the number of predictors is reduced by BIC
DEFAULT_filter_priors_for_clr = False


class BBSR(base_regression.BaseRegression):
    # Bayseian correlation measurements

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

    ols_only = False

    def __init__(self, X, Y, clr_mat, prior_mat, nS=DEFAULT_nS, prior_weight=DEFAULT_prior_weight,
                 no_prior_weight=DEFAULT_no_prior_weight, ordinary_least_squares=False):
        """
        Create a Regression object for Bayes Best Subset Regression

        :param X: Expression or Activity data [N x K]
        :type X: InferelatorData
        :param Y: Response expression data [N x G]
        :type Y: InferelatorData
        :param clr_mat: Calculated CLR between features of X & Y [G x K]
        :type clr_mat: pd.DataFrame
        :param prior_mat: Prior data between features of X & Y [G x K]
        :type prior_mat: pd.DataFrame

        :param nS: int
            Number of predictors to retain
        :param prior_weight: int
            Weight of a predictor which does have a prior
        :param no_prior_weight: int
            Weight of a predictor which doesn't have a prior
        """

        super(BBSR, self).__init__(X, Y)

        self.nS = nS
        self.ols_only = ordinary_least_squares

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

    def regress(self):
        """
        Execute BBSR

        :return: pd.DataFrame [G x K], pd.DataFrame [G x K]
            Returns the regression betas and beta error reductions for all threads if this is the master thread (rank 0)
            Returns None, None if it's a subordinate thread
        """

        if MPControl.is_dask():
            from inferelator.distributed.dask_functions import bbsr_regress_dask
            return bbsr_regress_dask(self.X, self.Y, self.pp, self.weights_mat, self.G, self.genes, self.nS)

        def regression_maker(j):
            level = 0 if j % 100 == 0 else 2
            utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=self.genes[j], i=j, total=self.G),
                                 level=level)

            data = bayes_stats.bbsr(self.X.values,
                                    utils.scale_vector(self.Y.get_gene_data(j, force_dense=True).flatten()),
                                    self.pp.iloc[j, :].values.flatten(),
                                    self.weights_mat.iloc[j, :].values.flatten(),
                                    self.nS,
                                    ordinary_least_squares=self.ols_only)
            data['ind'] = j
            return data

        return MPControl.map(regression_maker, range(self.G), tell_children=False)

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
    def _calculate_weight_matrix(p_matrix, no_p_weight=DEFAULT_no_prior_weight,
                                 p_weight=DEFAULT_prior_weight):
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


class BBSRRegressionWorkflowMixin(base_regression._RegressionWorkflowMixin):
    """
    Bayesian Best Subset Regression (BBSR)

    https://doi.org/10.15252/msb.20156236
    """

    mi_driver = mi.MIDriver
    mi_sync_path = None

    prior_weight = DEFAULT_prior_weight
    no_prior_weight = DEFAULT_no_prior_weight
    bsr_feature_num = DEFAULT_nS
    clr_only = False
    ols_only = False

    def set_regression_parameters(self, prior_weight=None, no_prior_weight=None, bsr_feature_num=None, clr_only=False,
                                  ordinary_least_squares_only=None):
        """
        Set regression parameters for BBSR

        :param prior_weight: Weight for edges that are present in the prior network. Defaults to 1.
        :type prior_weight: float
        :param no_prior_weight: Weight for edges that are not present in the prior network. Defaults to 1.
        :type no_prior_weight: float
        :param bsr_feature_num: The number of features to include in best subset regression. Defaults to 10.
        :type bsr_feature_num: int
        :param clr_only: Only use Context Likelihood of Relatedness to select features for BSR, not prior edges.
            Defaults to False.
        :type clr_only: bool
        :param ordinary_least_squares_only: Use OLS instead of Bayesian regression, for testing. Defaults to False.
        :type ordinary_least_squares_only: bool
        """

        self._set_with_warning("prior_weight", prior_weight)
        self._set_with_warning("no_prior_weight", no_prior_weight)
        self._set_with_warning("bsr_feature_num", bsr_feature_num)
        self._set_without_warning("clr_only", clr_only)
        self._set_without_warning("ols_only", ordinary_least_squares_only)

    def run_bootstrap(self, bootstrap):
        X = self.design.get_bootstrap(bootstrap)
        Y = self.response.get_bootstrap(bootstrap)

        utils.Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=0)
        clr_matrix, _ = self.mi_driver().run(Y, X, return_mi=False)
        utils.Debug.vprint('Calculating betas using BBSR', level=0)

        # Create a mock prior with no information if clr_only is set
        if self.clr_only:
            priors = pd.DataFrame(0, index=self.priors_data.index, columns=self.priors_data.columns)
        else:
            priors = self.priors_data

        return BBSR(X, Y, clr_matrix, priors, prior_weight=self.prior_weight,
                    no_prior_weight=self.no_prior_weight, nS=self.bsr_feature_num,
                    ordinary_least_squares=self.ols_only).run()
