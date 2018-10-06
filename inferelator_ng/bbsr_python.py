import pandas as pd
import numpy as np
from . import utils
from . import bayes_stats

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


class BBSR:
    # These are all the things that have to be set in a new BBSR class

    # Variables that handle multiprocessing via SLURM / KVS
    # The defaults here are placeholders for troubleshooting
    # All three of these should always be provided when instantiating
    kvs = None  # KVSClient
    chunk = None  # int

    # Raw Data
    X = None  # [K x N] float
    Y = None  # [G x N] float
    G = None  # int G
    K = None  # int K
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

    def __init__(self, X, Y, clr_mat, prior_mat, kvs, nS=DEFAULT_nS, prior_weight=DEFAULT_prior_weight,
                 no_prior_weight=DEFAULT_no_prior_weight, chunk=25):
        """
        Create a BBSR object for Bayes Best Subset Regression

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
        :param kvs: KVSClient
            KVS client object (for SLURM)
        """

        self.nS = nS
        self.kvs = kvs
        self.chunk = chunk

        # Calculate the weight matrix
        self.prior_weight = prior_weight
        self.no_prior_weight = no_prior_weight
        weights_mat = self._calculate_weight_matrix(prior_mat, p_weight=prior_weight, no_p_weight=no_prior_weight)
        utils.Debug.vprint("Weight matrix {} construction complete".format(weights_mat.shape))

        # Get the IDs and total count for the genes and predictors
        self.K = X.shape[0]
        self.tfs = X.index.values.tolist()
        self.G = Y.shape[0]
        self.genes = Y.index.values.tolist()

        # Rescale input data
        self.X = self._scale_and_permute(X)
        utils.Debug.vprint("Predictor matrix {} ready".format(X.shape))
        self.Y = self._scale_and_permute(Y)
        utils.Debug.vprint("Response matrix {} ready".format(Y.shape))

        # Rebuild weights, priors, and the CLR matrix for the features that are in this bootstrap
        self.weights_mat = weights_mat.loc[self.genes, self.tfs]
        self.prior_mat = prior_mat.loc[self.genes, self.tfs]
        self.clr_mat = clr_mat.loc[self.genes, self.tfs]
        utils.Debug.vprint("Selection of weights and priors {} complete".format(self.prior_mat.shape))

        # Build a boolean matrix indicating which tfs should be used as predictors for regression for each gene
        self.pp = self._build_pp_matrix()
        utils.Debug.vprint("Selection of predictors {} complete".format(self.pp.shape))

    def run(self):
        """
        Execute BBSR separately on each response variable in the data

        :return: pd.DataFrame [G x K], pd.DataFrame [G x K]
            Returns the regression betas and beta error reductions for all threads if this is the master thread (rank 0)
            Returns None, None if it's a subordinate thread
        """
        regression_data = []

        # For every response variable G, check to see if this thread should run BBSR for that variable
        # If it should (ownCheck is TRUE), slice the data for that response variable
        # And send the values (as an ndarray because pd.df indexing is glacially slow) to bayes_stats.bbsr
        # Keep a list of the resulting regression results
        oc = self.kvs.own_check(chunk=self.chunk)
        for j in range(self.G):
            if next(oc):
                level = 0 if j % 100 == 0 else 2
                utils.Debug.vprint("Regression on {gn} [{i} / {total}]".format(gn=self.Y.index[j],
                                                                               i=j,
                                                                               total=self.G), level=level)
                data = bayes_stats.bbsr(self.X.values,
                                        self.Y.ix[j, :].values,
                                        self.pp.ix[j, :].values,
                                        self.weights_mat.ix[j, :].values,
                                        self.nS)
                data['ind'] = j
                regression_data.append(data)

        # Put the regression results that this thread has calculated into KVS
        self.kvs.put('plist', (self.kvs.rank, regression_data))

        # If this is the master thread, pile the regression betas into dataframes and return them
        if self.kvs.is_master:
            return self.pileup_data()
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

    @staticmethod
    def _scale_and_permute(df):
        """
        Center and normalize a DataFrame
        :param df: pd.DataFrame
        :return df: pd.DataFrame
        """
        df = df.T
        return ((df - df.mean()) / df.std(ddof=1)).T

    def pileup_data(self):
        """
        Take the completed run data and pack it up into a DataFrame of betas
        :return: (pd.DataFrame [G x K], pd.DataFrame [G x K])
        """
        run_data = []

        # Reach into KVS to get the model data
        for p in range(utils.slurm_envs()['tasks']):
            pid, ps = self.kvs.get('plist')
            run_data.extend(ps)
            utils.Debug.vprint("Collected {l} models from proc {id}".format(l=len(ps), id=pid), level=2)
        self.kvs.finish_own_check()

        # Create G x K arrays of 0s to populate with the regression data
        betas = np.zeros((self.G, self.K), dtype=np.dtype(float))
        betas_rescale = np.zeros((self.G, self.K), dtype=np.dtype(float))

        # Populate the zero arrays with the BBSR betas
        for data in run_data:
            xidx = data['ind']  # Int
            yidx = data['pp']  # Boolean array of size K

            betas[xidx, yidx] = data['betas']
            betas_rescale[xidx, yidx] = data['betas_resc']

        d_len, b_avg, null_m = self._summary_stats(betas)
        utils.Debug.vprint("Regression complete:", end=" ", level=0)
        utils.Debug.vprint("{d_len} Models, {b_avg} Preds per Model, {nom} Null Models".format(d_len=d_len,
                                                                                               b_avg=round(b_avg, 4),
                                                                                               nom=null_m), level=0)

        # Convert arrays into pd.DataFrames to return results
        betas = pd.DataFrame(betas, index=self.Y.index, columns=self.X.index)
        betas_rescale = pd.DataFrame(betas_rescale, index=self.Y.index, columns=self.X.index)

        return betas, betas_rescale

    @staticmethod
    def _summary_stats(arr):
        d_len = arr.shape[0]
        b_avg = np.mean(np.sum(arr != 0, axis=1))
        null_m = np.sum(np.sum(arr != 0, axis=1) == 0)
        return d_len, b_avg, null_m


class BBSR_runner:
    """
    Wrapper for the BBSR class. Passes arguments in and then calls run. Returns the result.
    """

    def run(self, X, Y, clr, prior_mat, kvs=None, rank=0, ownCheck=None):
        return BBSR(X, Y, clr, prior_mat, kvs).run()
