import numpy as np
import pandas as pd
import copy

from inferelator import utils, default
from inferelator.distributed.inferelator_mp import MPControl
from inferelator.utils import Validator as check

DEFAULT_CHUNK = 25
PROGRESS_STR = "Regression on {gn} [{i} / {total}]"


class BaseRegression(object):
    # These are all the things that have to be set in a new regression class

    chunk = DEFAULT_CHUNK  # int

    # Raw Data
    X = None  # [K x N] float
    Y = None  # [G x N] float
    G = None  # int G
    K = None  # int K

    def __init__(self, X, Y):
        """
        Create a regression object and do basic data transforms

        :param X: pd.DataFrame [K x N]
            Expression / Activity data
        :param Y: pd.DataFrame [G x N]
            Response data
        """

        # Get the IDs and total count for the genes and predictors
        self.K = X.shape[0]
        self.tfs = X.index.values.tolist()
        self.G = Y.shape[0]
        self.genes = Y.index.values.tolist()

        # Rescale input data
        self.X = self._scale(X)
        self.Y = self._scale(Y)
        utils.Debug.vprint("Predictor matrix {pr} and response matrix {re} ready".format(pr=X.shape,
                                                                                         re=Y.shape))

    def run(self):
        """
        Execute regression separately on each response variable in the data

        :return: pd.DataFrame [G x K], pd.DataFrame [G x K]
            Returns the regression betas and beta error reductions for all threads if this is the master thread (rank 0)
            Returns None, None if it's a subordinate thread
        """

        run_data = self.regress()

        if MPControl.is_master:
            pileup_data = self.pileup_data(run_data)
        else:
            pileup_data = None, None

        MPControl.sync_processes("post_pileup")
        return pileup_data

    def regress(self):
        """
        Execute regression and return a list which can be provided to pileup_data
        :return: list
        """
        raise NotImplementedError

    @staticmethod
    def _scale(df):
        """
        Center and normalize a DataFrame
        :param df: pd.DataFrame
        :return df: pd.DataFrame
        """
        df = df.T
        return ((df - df.mean()) / df.std(ddof=1)).T

    def pileup_data(self, run_data):
        """
        Take the completed run data and pack it up into a DataFrame of betas

        :param run_data: list
            A list of regression result dicts ordered by gene. Each regression result should have `ind`, `pp`, `betas`
            and `betas_resc` keys with the appropriate data.
        :return betas, betas_rescale: (pd.DataFrame [G x K], pd.DataFrame [G x K])
        """

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
        utils.Debug.vprint("{d_len} Models, {b_avg} Preds per Model ({nom} Null)".format(d_len=d_len,
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


class RegressionWorkflow(object):
    """
    RegressionWorkflow implements run_regression and run_bootstrap
    Each regression method needs to extend this to implement run_bootstrap (and also run_regression if necessary)
    """
    random_seed = default.DEFAULT_RANDOM_SEED
    num_bootstraps = default.DEFAULT_NUM_BOOTSTRAPS

    def run_regression(self):
        betas = []
        rescaled_betas = []

        MPControl.sync_processes("pre_regression")

        for idx, bootstrap in enumerate(self.get_bootstraps()):
            utils.Debug.vprint('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps), level=0)
            np.random.seed(self.random_seed + idx)
            current_betas, current_rescaled_betas = self.run_bootstrap(bootstrap)
            if self.is_master():
                betas.append(current_betas)
                rescaled_betas.append(current_rescaled_betas)

            MPControl.sync_processes("post_bootstrap")

        return betas, rescaled_betas

    def run_bootstrap(self, bootstrap):
        raise NotImplementedError


def recalculate_betas_from_selected(x, y, idx=None):
    """
    Estimate betas from a selected subset of predictors
    :param x: np.ndarray [N x k]
        Predictor matrix
    :param y: np.ndarray [N x 1]
        Response matrix
    :param idx: np.ndarray [k x 1]
        Predictors to use (unused predictors will return a beta of 0)
        If None, use all predictors
    :return: np.ndarray [k,]
        Estimated beta-hats
    """

    # Create an array of size [k,] to hold the estimated betas
    best_betas = np.zeros(x.shape[1], dtype=np.dtype(float))

    # Use all predictors if no subset index is passed in
    if idx is None:
        idx = np.ones(x.shape[1], dtype=np.dtype(bool))

    # Convert boolean array to an array of indexes
    idx = bool_to_index(idx)

    # Subset the predictors with the index array
    x = x[:, idx]

    # Solve for beta-hat with LAPACK or return a null model if xTx is singular
    xtx = np.dot(x.T, x)
    if np.linalg.matrix_rank(xtx) == xtx.shape[1]:
        beta_hat = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
    else:
        beta_hat = np.zeros(len(idx), dtype=np.dtype(float))

    # Use the index array to write beta-hats
    # This yields the same size result matrix as number of predictors in x
    # (even if x is subset with an index)
    for i, j in enumerate(idx):
        best_betas[j] = beta_hat[i]
    return best_betas


def predict_error_reduction(x, y, betas):
    """
    Predict the error reduction from each predictor
    :param x: np.ndarray [n x k]
    :param y: np.ndarray [n x 1]
    :param betas: np.ndarray [k x 1]
    :return: np.ndarray [k,]
    """
    assert check.argument_type(betas, np.ndarray)

    (n, k) = x.shape
    pp_idx = index_of_nonzeros(betas).tolist()

    # Calculate the variance of the residuals
    ss_all = sigma_squared(x, y, betas)
    error_reduction = np.zeros(k, dtype=np.dtype(float))

    if len(pp_idx) == 1:
        error_reduction[pp_idx] = 1 - (ss_all / np.var(y, ddof=1))
        return error_reduction

    for pp_i in range(len(pp_idx)):
        # Copy the index of predictors
        leave_out = copy.copy(pp_idx)
        # Pull off one of the predictors
        lost = leave_out.pop(pp_i)

        # Reestimate betas for all the predictors except the one that we removed
        x_leaveout = x[:, leave_out]
        try:
            beta_hat = np.linalg.solve(np.dot(x_leaveout.T, x_leaveout), np.dot(x_leaveout.T, y))
        except np.linalg.LinAlgError:
            beta_hat = np.zeros(len(leave_out), dtype=np.dtype(float))

        # Calculate the variance of the residuals for the new estimated betas
        ss_leaveout = sigma_squared(x_leaveout, y, beta_hat)
        error_reduction[lost] = 1 - (ss_all / ss_leaveout)

    return error_reduction


def sigma_squared(x, y, betas):
    return np.var(np.subtract(y, np.dot(x, betas).reshape(-1, 1)), ddof=1)


def index_of_nonzeros(arr):
    """
    Returns an array that indexes all the non-zero elements of an array
    :param arr: np.ndarray
    :return: np.ndarray
    """
    return np.where(arr != 0)[0]


def bool_to_index(arr):
    """
    Returns an array that indexes all the True elements of a boolean array
    :param arr: np.ndarray
    :return: np.ndarray
    """
    assert check.argument_type(arr, np.ndarray)
    return np.where(arr)[0]
