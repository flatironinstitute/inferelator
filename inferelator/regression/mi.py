from __future__ import division

import numpy as np
import pandas as pd
import scipy.sparse as sps

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.utils import Debug, InferelatorData, array_set_diag
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

    @staticmethod
    def run(x, y, bins=DEFAULT_NUM_BINS, logtype=DEFAULT_LOG_TYPE, return_mi=True):
        return context_likelihood_mi(x, y, bins=bins, logtype=logtype, return_mi=return_mi)


def context_likelihood_mi(x, y, bins=DEFAULT_NUM_BINS, logtype=DEFAULT_LOG_TYPE, return_mi=True):
    """
    Wrapper to calculate the Context Likelihood of Relatedness and Mutual Information for two data sets that have
    common condition rows. The y argument will be used to calculate background MI for the x & y MI.
    As an implementation detail, y will be cast to a dense array if it is sparse.
    X can be sparse with no internal copy.

    This function handles unpacking and packing the InferelatorData.

    :param x: An N x G InferelatorData object
    :type x: InferelatorData [N x G]
    :param y: An N x K InferelatorData object
    :type y: InferelatorData [N x K]
    :param logtype: The logarithm function to use when calculating information. Defaults to natural log (np.log)
    :type logtype: np.log func
    :param bins: Number of bins for discretizing continuous variables
    :type bins: int
    :param return_mi: Boolean for returning a MI object. Defaults to True
    :type return_mi: bool
    :return clr, mi: CLR and MI InferelatorData objects. Returns (CLR, None) if return_mi is False.
    :rtype InferelatorData, InferelatorData:
    """

    assert check.argument_integer(bins, allow_none=True)
    assert min(x.shape) > 0
    assert min(y.shape) > 0
    assert check.indexes_align((x.sample_names, y.sample_names))

    # Create dense output matrix and copy the inputs
    mi_r = x.gene_names
    mi_c = y.gene_names

    # Build a [G x K] mutual information array
    mi = mutual_information(x.expression_data, y.expression_data, bins, logtype=logtype)
    array_set_diag(mi, 0., mi_r, mi_c)

    # Build a [K x K] mutual information array
    mi_bg = mutual_information(y.expression_data, y.expression_data, bins, logtype=logtype)
    array_set_diag(mi_bg, 0., mi_c, mi_c)

    # Calculate CLR
    clr = calc_mixed_clr(mi, mi_bg)

    MPControl.sync_processes(pref=SYNC_CLR_KEY)

    mi = pd.DataFrame(mi, index=mi_r, columns=mi_c)
    clr = pd.DataFrame(clr, index=mi_r, columns=mi_c)

    return clr, mi if return_mi else None


def mutual_information(x, y, bins, logtype=DEFAULT_LOG_TYPE):
    """
    Calculate the mutual information matrix between two data matrices, where the columns are equivalent conditions

    :param x: np.array (n x m1)
        The data from m1 variables across n conditions
    :param y: np.array (n x m2)
        The data from m2 variables across n conditions
    :param bins: int
        Number of bins to discretize continuous data into for the generation of a contingency table
    :param logtype: np.log func
        Which type of log function should be used (log2 results in MI bits, log results in MI nats, log10... is weird)

    :return mi: pd.DataFrame (m1 x m2)
        The mutual information between variables m1 and m2
    """

    # Discretize the input matrix y
    y = y.A if sps.isspmatrix(y) else y
    y = _make_array_discrete(y, bins, axis=0)

    # Build the MI matrix
    if MPControl.is_dask():
        from inferelator.distributed.dask_functions import build_mi_array_dask
        return build_mi_array_dask(x, y, bins, logtype=logtype)
    else:
        return build_mi_array(x, y, bins, logtype=logtype)


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
        Debug.allprint("Mutual Information Calculation [{i} / {total}]".format(i=i, total=m1), level=level)

        discrete_X = _make_discrete(X[:, i].A.flatten() if sps.isspmatrix(X) else X[:, i].flatten(), bins)
        return [_calc_mi(_make_table(discrete_X, Y[:, j], bins), logtype=logtype) for j in range(m2)]

    # Send the MI build to the multiprocessing controller
    mi_list = MPControl.map(mi_make, range(m1), tmp_file_path=temp_dir)

    # Convert the list of lists to an array
    mi = np.array(mi_list)
    assert (m1, m2) == mi.shape, "Array {sh} produced [({m1}, {m2}) expected]".format(sh=mi.shape, m1=m1, m2=m2)

    return mi


def calc_mixed_clr(mi, mi_bg):
    """
    Calculate the context liklihood of relatedness from mutual information and the background mutual information

    :param mi: Mutual information array [m1 x m2]
    :type mi: np.ndarray
    :param mi_bg: Background mutual information array [m2 x m2]
    :type mi_bg: np.ndarray
    :return clr: Context liklihood of relateness array [m1 x m2]
    :rtype: np.ndarray
    """

    with np.errstate(invalid='ignore'):

        # Calculate the zscore for the dynamic CLR
        z_dyn = np.round(mi, 10)  # Rounding so that float precision differences don't turn into huge CLR differences
        z_dyn = np.subtract(z_dyn, np.mean(mi, axis=0))
        z_dyn = np.divide(z_dyn, np.std(mi, axis=0, ddof=CLR_DDOF))

        # Calculate the zscore for the static CLR
        z_stat = np.round(mi, 10)  # Rounding so that float precision differences don't turn into huge CLR differences
        z_stat = np.subtract(z_stat, np.mean(mi_bg, axis=0))
        z_stat = np.divide(z_stat, np.std(mi_bg, axis=0, ddof=CLR_DDOF))

        z_dyn[z_dyn < 0] = 0
        z_stat[z_stat < 0] = 0

    # Calculate CLR
    return np.sqrt(np.square(z_dyn) + np.square(z_stat))


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
    assert check.argument_type(arr_vec, np.ndarray)
    assert len(arr_vec.shape) == 1 or arr_vec.shape[1] == 1

    # Create a function to convert continuous values to discrete bins
    arr_min = np.min(arr_vec)
    arr_max = np.max(arr_vec)

    if arr_min == arr_max:
        return np.zeros(shape=arr_vec.shape, dtype=np.dtype(int))

    try:
        eps = np.finfo(arr_vec.dtype).eps
    except ValueError:
        eps = np.finfo(float).eps

    eps_mod = max(eps, eps * (arr_max - arr_min))

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
    with np.errstate(divide='ignore', invalid='ignore'):
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
        return np.sum(mi_val, axis=(0, 1))
