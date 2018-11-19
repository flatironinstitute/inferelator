import numpy as np
import pandas as pd
import copy

from inferelator_ng import utils


class BaseRegression:
    # These are all the things that have to be set in a new regression class

    # Variables that handle multiprocessing via SLURM / KVS
    # The defaults here are placeholders for troubleshooting
    # These should always be provided when instantiating
    kvs = None  # KVSClient
    chunk = None  # int

    # Raw Data
    X = None  # [K x N] float
    Y = None  # [G x N] float
    G = None  # int G
    K = None  # int K

    def run(self):
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
        self.kvs.master_remove_key()

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


def recalculate_betas_from_selected(x, y, idx):
    """
    Estimate betas from a selected subset of predictors
    :param x: np.ndarray [n x k]
    :param y: np.ndarray [n x 1]
    :param idx: np.ndarray [k x 1]
    :return: np.ndarray [k,]
    """
    best_betas = np.zeros(x.shape[1], dtype=np.dtype(float))
    idx = bool_to_index(idx)
    x = x[:, idx]
    beta_hat = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
    for i, j in enumerate(idx):
        best_betas[j] = beta_hat[i]
    return best_betas


def predict_error_reduction(x, y, betas):
    """
    Predict the error reduction from each predictor
    :param x: np.ndarray [n x k]
    :param y: np.ndarray [n x 1]
    :param betas: list [k]
    :return: np.ndarray [k,]
    """
    (n, k) = x.shape
    pp_idx = index_of_nonzeros(betas).tolist()

    ss_all = sigma_squared(x, y, betas)
    error_reduction = np.zeros(k, dtype=np.dtype(float))

    if len(pp_idx) == 1:
        error_reduction[pp_idx] = 1 - (ss_all / np.var(y, ddof=1))
        return error_reduction

    for pp_i in range(len(pp_idx)):
        leave_out = copy.copy(pp_idx)
        lost = leave_out.pop(pp_i)

        x_leaveout = x[:, leave_out]
        beta_hat = np.linalg.solve(np.dot(x_leaveout.T, x_leaveout), np.dot(x_leaveout.T, y))

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
    return np.where(arr)[0]
