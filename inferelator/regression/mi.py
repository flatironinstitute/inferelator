from __future__ import division

import os
import numpy as np
import pandas as pd

from inferelator.distributed.inferelator_mp import MPControl
from inferelator import utils
from inferelator.utils import Validator as check

# Number of discrete bins for mutual information calculation
DEFAULT_NUM_BINS = 10

# DDOF for CLR
CLR_DDOF = 1

# Log type for MI calculations. np.log2 gives results in bits; np.log gives results in nats
DEFAULT_LOG_TYPE = np.log

# KVS keys for multiprocessing
SYNC_CLR_KEY = 'post_clr'


class MIDriver:
    bins = DEFAULT_NUM_BINS

    def __init__(self, bins=DEFAULT_NUM_BINS, sync_in_tmp_path=None):
        """
        :param bins: int
            Number of bins for discretizing continuous variables
        :param sync_in_tmp_path: path
            Path to a temp file directory to use for synchronizing processes
            This uses the temp directory for DATA only. Process communication is still done with KVS.
        """

        assert check.argument_path(sync_in_tmp_path, allow_none=True, access=os.W_OK)
        assert check.argument_integer(bins, allow_none=False)

        self.bins = bins
        self.temp_dir = sync_in_tmp_path

    def run(self, x_df, y_df, bins=None, logtype=DEFAULT_LOG_TYPE):
        """
        Wrapper to calculate the CLR and MI for two data sets that have common condition columns
        :param x_df: pd.DataFrame
        :param y_df: pd.DataFrame
        :param logtype: np.log func
        :param bins: int
            Number of bins for discretizing continuous variables
        :return clr, mi: pd.DataFrame, pd.DataFrame
            CLR and MI DataFrames
        """

        assert check.argument_integer(bins, allow_none=True)
        assert check.indexes_align((x_df.columns, y_df.columns))
        assert x_df.shape[0] > 0
        assert x_df.shape[1] > 0
        assert y_df.shape[0] > 0
        assert y_df.shape[1] > 0

        if bins is not None:
            self.bins = bins

        mi = mutual_information(y_df, x_df, self.bins, temp_dir=self.temp_dir, logtype=logtype)
        mi_bg = mutual_information(x_df, x_df, self.bins, temp_dir=self.temp_dir, logtype=logtype)
        clr = calc_mixed_clr(utils.df_set_diag(mi, 0), utils.df_set_diag(mi_bg, 0))

        MPControl.sync_processes(pref=SYNC_CLR_KEY)

        return clr, mi


def mutual_information(X, Y, bins, logtype=DEFAULT_LOG_TYPE, temp_dir=None):
    """
    Calculate the mutual information matrix between two data matrices, where the columns are equivalent conditions

    :param X: pd.DataFrame (m1 x n)
        The data from m1 variables across n conditions
    :param Y: pd.DataFrame (m2 x n)
        The data from m2 variables across n conditions
    :param bins: int
        Number of bins to discretize continuous data into for the generation of a contingency table
    :param logtype: np.log func
        Which type of log function should be used (log2 results in MI bits, log results in MI nats, log10... is weird)
    :param temp_dir: path
        Path to write temp files for multiprocessing

    :return mi: pd.DataFrame (m1 x m2)
        The mutual information between variables m1 and m2
    """

    assert check.indexes_align((X.columns, Y.columns))

    # Create dense output matrix and copy the inputs
    mi_r = X.index
    mi_c = Y.index

    X = X.values
    Y = Y.values

    # Discretize the input matrixes
    X = _make_array_discrete(X.transpose(), bins, axis=0)
    Y = _make_array_discrete(Y.transpose(), bins, axis=0)

    # Build the MI matrix
    if MPControl.is_dask():
        from inferelator.distributed.dask_functions import build_mi_array_dask
        return pd.DataFrame(build_mi_array_dask(X, Y, bins, logtype=logtype), index=mi_r, columns=mi_c)
    else:
        return pd.DataFrame(build_mi_array(X, Y, bins, logtype=logtype, temp_dir=temp_dir), index=mi_r,
                            columns=mi_c)


def build_mi_array(X, Y, bins, logtype=DEFAULT_LOG_TYPE, temp_dir=None):
    """
    Calculate MI into an array

    :param X: np.ndarray (n x m1)
        Discrete array of bins
    :param Y: np.ndarray (n x m2)
        Discrete array of bins
    :param bins: int
        The total number of bins that were used to make the arrays discrete
    :param logtype: np.log func
        Which log function to use (log2 gives bits, ln gives nats)
    :param temp_dir: path
        Path to write temp files for multiprocessing
    :return mi: np.ndarray (m1 x m2)
        Returns the mutual information array
    """

    m1, m2 = X.shape[1], Y.shape[1]

    # Define the function which calculates MI for each variable in X against every variable in Y
    def mi_make(i):
        level = 2 if i % 1000 == 0 else 3
        utils.Debug.allprint("Mutual Information Calculation [{i} / {total}]".format(i=i, total=m1), level=level)
        return [_calc_mi(_make_table(X[:, i], Y[:, j], bins), logtype=logtype) for j in range(m2)]

    # Send the MI build to the multiprocessing controller
    mi_list = MPControl.map(mi_make, range(m1), tmp_file_path=temp_dir)

    # Convert the list of lists to an array
    mi = np.array(mi_list)
    assert (m1, m2) == mi.shape, "Array {sh} produced [({m1}, {m2}) expected]".format(sh=mi.shape, m1=m1, m2=m2)

    return mi


def calc_mixed_clr(mi, mi_bg):
    """
    Calculate the context liklihood of relatedness from mutual information and the background mutual information

    :param mi: pd.DataFrame
        Mutual information dataframe
    :param mi_bg: pd.DataFrame
        Background mutual information dataframe
    :return clr: pd.DataFrame
        Context liklihood of relateness dataframe
    """
    # Calculate the zscore for columns
    z_col = mi.copy().round(10)  # Rounding so that float precision differences don't turn into huge CLR differences
    z_col = z_col.subtract(mi_bg.mean(axis=0))
    z_col = z_col.divide(mi_bg.std(axis=0, ddof=CLR_DDOF))
    z_col[z_col < 0] = 0

    # Calculate the zscore for rows
    z_row = mi.copy().round(10)  # Rounding so that float precision differences don't turn into huge CLR differences
    z_row = z_row.subtract(mi.mean(axis=1), axis=0)
    z_row = z_row.divide(mi.std(axis=1, ddof=CLR_DDOF), axis=0)
    z_row[z_row < 0] = 0

    clr = np.sqrt(np.square(z_col) + np.square(z_row))
    return clr


def _make_array_discrete(array, num_bins, axis=0):
    """
    Applies _make_discrete to a 2d array
    """
    return np.apply_along_axis(_make_discrete, arr=array, axis=axis, num_bins=num_bins)


def _make_discrete(arr_vec, num_bins):
    """
    Takes a 1d array or vector and discretizes it into nonparametric bins
    :param arr_vec: np.ndarray
        1d array of continuous data
    :param num_bins: int
        Number of bins for data
    :return array: np.ndarray
        1d array of discrete data
    """

    # Sanity check
    if not isinstance(arr_vec, list):
        try:
            if arr_vec.shape[1] != 1:
                raise ValueError("make_discrete takes a 1d array")
        except IndexError:
            pass

    # Create a function to convert continuous values to discrete bins
    arr_min = np.min(arr_vec)
    arr_max = np.max(arr_vec)
    eps_mod = max(np.finfo(float).eps, np.finfo(float).eps * (arr_max - arr_min))

    def _disc_func(x):
        return np.floor((x - arr_min) / (arr_max - arr_min + eps_mod) * num_bins)

    # Apply the function to every value in the vector
    return _disc_func(arr_vec).astype(np.dtype(int))


def _make_table(x, y, num_bins):
    """
    Takes two variable vectors which have been made into discrete integer bins and constructs a contingency table
    :param x: np.ndarray
        1d array of discrete data
    :param y: np.ndarray
        1d array of discrete data
    :param num_bins: int
        Number of bins for data
    :return ctable: np.ndarray (num_bins x num_bins)
        Contingency table of variables X and Y
    """

    assert len(x.shape) == 1
    assert len(y.shape) == 1

    # The only fast way to do this is by reindexing the table as an index array
    reindex = x * num_bins + y
    # Then piling everything up with bincount and reshaping it back into the table
    return np.bincount(reindex, minlength=num_bins ** 2).reshape(num_bins, num_bins).astype(np.dtype(float))


def _calc_mi(table, logtype=DEFAULT_LOG_TYPE):
    """
    Calculate Mutual Information from a contingency table of two variables

    :param table: np.ndarray (num_bins x num_bins)
        Contingency table
    :param logtype: np.log func
        Log function to use
    :return: float
        Mutual information between variable x & y
    """

    # Turn off runtime warnings (there is an explicit check for NaNs and INFs in-function)
    reset = np.seterr(divide='ignore', invalid='ignore')

    m, n = table.shape
    assert n == m

    total = np.sum(table, axis=(0, 1))

    # (PxPy) [n x n]
    mi_val = np.dot((np.sum(table, axis=1) / total).reshape(-1, 1),
                    (np.sum(table, axis=0) / total).reshape(1, -1))

    # (Pxy) [n x n]
    table = np.divide(table, total)

    # (Pxy)/(PxPy) [n x n]
    mi_val = np.divide(table, mi_val)

    # log[(Pxy)/(PxPy)] [n x n]
    mi_val = logtype(mi_val)

    # Pxy(log[(Pxy)/(PxPy)]) [n x n]
    mi_val = np.multiply(table, mi_val)
    mi_val[np.isnan(mi_val)] = 0

    # Summation
    mi_val = np.sum(mi_val, axis=(0, 1))

    np.seterr(**reset)
    return mi_val
