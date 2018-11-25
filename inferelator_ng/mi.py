import itertools
import tempfile
import os
import numpy as np
import pandas as pd
from inferelator_ng import utils

# Number of discrete bins for mutual information calculation
DEFAULT_NUM_BINS = 10

# DDOF for CLR
CLR_DDOF = 1

# Log type for MI calculations. np.log2 gives results in bits; np.log gives results in nats
DEFAULT_LOG_TYPE = np.log

# Multiprocessing chunk size
DEFAULT_CHUNK = 2000

# KVS keys for multiprocessing
COUNT_KEY = 'micount'
PILEUP_DATA_KEY = 'mi_pileup'
FINAL_MI_DATA_KEY = 'mi_final'
SYNC_CLR_KEY = 'post_clr'
SYNC_MI_KEY = 'post_mi'


class MIDriver:
    bins = DEFAULT_NUM_BINS
    kvs = None

    def __init__(self, bins=DEFAULT_NUM_BINS, kvs=None, sync_in_tmp_path=None):
        """
        :param bins: int
            Number of bins for discretizing continuous variables
        :param kvs: KVSController
            KVSController object for interprocess communication
        :param sync_in_tmp_path: path
            Path to a temp file directory to use for synchronizing processes
            This uses the temp directory for DATA only. Process communication is still done with KVS.
            If KVS is None, this does nothing.
        """

        if sync_in_tmp_path is not None:
            assert os.access(sync_in_tmp_path, os.W_OK), "Directory {di} not writable".format(di=sync_in_tmp_path)

        self.bins = bins
        self.kvs = kvs
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

        if bins is not None:
            self.bins = bins

        mi = mutual_information(y_df, x_df, self.bins, kvs=self.kvs, temp_dir=self.temp_dir, logtype=logtype)
        mi_bg = mutual_information(x_df, x_df, self.bins, kvs=self.kvs, temp_dir=self.temp_dir, logtype=logtype)
        clr = calc_mixed_clr(utils.df_set_diag(mi, 0), utils.df_set_diag(mi_bg, 0))

        if self.kvs is not None:
            self.kvs.sync_processes(pref=SYNC_CLR_KEY)

        return clr, mi


def mutual_information(X, Y, bins, logtype=DEFAULT_LOG_TYPE, kvs=None, temp_dir=None, chunk=DEFAULT_CHUNK):
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

    :return mi: pd.DataFrame (m1 x m2)
        The mutual information between variables m1 and m2
    """
    assert X.shape[1] == Y.shape[1]
    assert (X.columns == Y.columns).all()

    # Create dense output matrix and copy the inputs
    mi_r = X.index
    mi_c = Y.index

    X = X.values
    Y = Y.values

    # Discretize the input matrixes
    X = _make_array_discrete(X, bins, axis=1).transpose()
    Y = _make_array_discrete(Y, bins, axis=1).transpose()

    # If there is no KVS object, just run locally on one core
    if kvs is None:
        mi = build_mi_array(X, Y, bins, logtype=logtype)
    # If there is a KVS object, run distributed and give the results to everyone
    elif temp_dir is None:
        mi = mi_through_kvs(X, Y, bins, kvs, logtype=logtype, chunk=chunk)
    else:
        mi = mi_through_dir(X, Y, bins, kvs, temp_dir, logtype=logtype, chunk=chunk)

    return pd.DataFrame(mi, index=mi_r, columns=mi_c)


def mi_through_kvs(X, Y, bins, kvs, logtype=DEFAULT_LOG_TYPE, chunk=DEFAULT_CHUNK):
    # Run MI calculations on everything that an ownCheck gives to this process and stash it in KVS
    oc = kvs.own_check(chunk=chunk, kvs_key=COUNT_KEY)
    kvs.put(PILEUP_DATA_KEY, build_mi_array(X, Y, bins, logtype=logtype, oc=oc))

    # Block here until mi pileup is complete and then get the final mi matrix from mi_final
    if kvs.is_master:
        mi = np.full((X.shape[1], Y.shape[1]), np.nan, dtype=np.dtype(float))
        for _ in range(kvs.tasks):
            mi_two = kvs.get(PILEUP_DATA_KEY)
            update = ~np.isnan(mi_two)
            mi[update] = mi_two[update]
        kvs.put(FINAL_MI_DATA_KEY, mi)
    else:
        mi = kvs.view(FINAL_MI_DATA_KEY)

    # Block here until all the processes have mi_final and then tear down the KVS data
    kvs.sync_processes(pref=SYNC_MI_KEY)
    kvs.master_remove_key(kvs_key=COUNT_KEY)
    kvs.master_remove_key(kvs_key=FINAL_MI_DATA_KEY)

    return mi


def mi_through_dir(X, Y, bins, kvs, temp_dir, logtype=DEFAULT_LOG_TYPE, chunk=DEFAULT_CHUNK):
    # Do MI calculations locally
    local_mi = build_mi_array(X, Y, bins, logtype=logtype, oc=kvs.own_check(chunk=chunk, kvs_key=COUNT_KEY))

    # Write these MI calculations to a temp file and put that filename on KVS
    temp_fd, temp_name = tempfile.mkstemp(prefix="mi", dir=temp_dir)
    with os.fdopen(temp_fd, "wb") as temp:
        np.savetxt(temp, local_mi, delimiter="\t")
    kvs.put(PILEUP_DATA_KEY, temp_name)

    # Pile up the resulting data
    if kvs.is_master:
        # Read in each temp file, put it into a master MI array, and then delete the temp file
        mi = np.full((X.shape[1], Y.shape[1]), np.nan, dtype=np.dtype(float))
        for _ in range(kvs.tasks):
            mi_tmp_path = kvs.get(PILEUP_DATA_KEY)
            with open(mi_tmp_path, mode="r") as temp:
                mi_two = np.loadtxt(temp, delimiter="\t")
            os.remove(mi_tmp_path)
            update = ~np.isnan(mi_two)
            mi[update] = mi_two[update]

        # Write the complete MI array to a temp file and put that filename on KVS
        final_fd, final_name = tempfile.mkstemp(prefix="mi", dir=temp_dir)
        with os.fdopen(final_fd, "wb") as temp:
            np.savetxt(temp, mi, delimiter="\t")
        kvs.put(FINAL_MI_DATA_KEY, final_name)
    # Get the complete MI array filename from KVS and read it in
    else:
        mi_tmp_path = kvs.view(FINAL_MI_DATA_KEY)
        with open(mi_tmp_path, mode="r") as temp:
            mi = np.loadtxt(temp, delimiter="\t")

    # Block here until all the processes have mi_final and then tear down the KVS data
    kvs.sync_processes(pref=SYNC_MI_KEY)
    kvs.master_remove_key(kvs_key=COUNT_KEY)
    kvs.master_remove_key(kvs_key=FINAL_MI_DATA_KEY)

    # Delete the complete MI array temp file
    if kvs.is_master:
        os.remove(final_name)

    return mi


def build_mi_array(X, Y, bins, logtype=DEFAULT_LOG_TYPE, oc=None):
    """
    Calculate MI into an array initialized with NaNs

    :param X: np.ndarray (n x m1)
        Discrete array of bins
    :param Y: np.ndarray (n x m2)
        Discrete array of bins
    :param bins: int
        The total number of bins that were used to make the arrays discrete
    :param logtype: np.log func
        Which log function to use (log2 gives bits, ln gives nats)
    :param oc: ownCheck generator
        The multiprocessing controller from KVS. Tells this process what to calculate.
        If None, calculate the entire array
    :return mi: np.ndarray (m1 x m2)
        Returns the mutual information calculated by this process.
        If oc isn't None, this array will be np.NaNs for all values not calculated
    """
    m1, m2 = X.shape[1], Y.shape[1]
    mi = np.full((m1, m2), np.nan, dtype=np.dtype(float))
    for i, j in itertools.product(range(m1), range(m2)):
        if oc is None or next(oc):
            ctable = _make_table(X[:, i], Y[:, j], bins)
            mi[i, j] = _calc_mi(ctable, logtype=logtype)
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


def _make_discrete(array, num_bins):
    """
    Takes a 1d array or vector and discretizes it into nonparametric bins
    :param array: np.ndarray
        1d array of continuous data
    :param num_bins: int
        Number of bins for data
    :return array: np.ndarray
        1d array of discrete data
    """

    # Sanity check
    if not isinstance(array, list):
        try:
            if array.shape[1] != 1:
                raise ValueError("make_discrete takes a 1d array")
        except IndexError:
            pass

    # Create a function to convert continuous values to discrete bins
    arr_min = np.min(array)
    arr_max = np.max(array)
    eps_mod = max(np.finfo(float).eps, np.finfo(float).eps * (arr_max - arr_min))
    disc_func = np.vectorize(lambda x: np.floor((x - arr_min) / (arr_max - arr_min + eps_mod) * num_bins))

    # Apply the function to every value in the vector
    return disc_func(array).astype(np.dtype(int))


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
