import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from . import utils

DDOF_default = 1
DEFAULT_method = 'ward'
DEFAULT_max_cluster_ratio = 0.25
DEFAULT_max_group_size = 0.1
DEFAULT_log_transform = np.log10

DEFAULT_start_position = 1


def initial_clustering(data):
    """
    Take single-cell expression data as a count matrix and cluster the cells by similarity
    Distance metric is 1 - Pearson correlation between cells
    :param data: pd.DataFrame [m x n]
        Dataframe of n single cells with m genes. Preferably in a count matrix
    :return: pd.DataFrame [m x c], np.ndarray [n, ]
        Returns bulked up counts for c clusters and a 1d array of indexes to convert the original dataframe to the
        clustered dataframe (or back).
    """

    # Convert the data to floats in a ndarray
    genes = data.index.values.tolist()
    norm_data = data.values.astype(np.float64)

    # Normalize the data for library size (per cell) and interval (per gene)
    ss_normalization(norm_data)
    utils.Debug.vprint("Interval and Library Size Normalization Complete [{}]".format(data.shape))

    # Calculate the distance matrix
    dist = 1 - _pearson_correlation_matrix(norm_data.T)
    utils.Debug.vprint("Distance matrix construction complete [{}]".format(dist.shape))

    # Convert the distance matrix to a squareform vector
    dist = squareform(dist, force='tovector', checks=False)

    # Perform clustering and find the optimal cluster cut using the default parameters above
    clust_idx = _find_optimal_cluster_cut(dist)

    # Bulk up the clusters
    return reclustering(data.values.astype(np.float64), clust_idx, index=genes), clust_idx


def declustering(c_data, clust_idx, columns=None):
    """
    Take bulked up data and break it into single cells again
    :param c_data: pd.DataFrame [m x c]
    :param clust_idx: np.ndarray [n, ]
    :param columns: list [n]
    :return: pd.DataFrame [m x n]
    """
    if columns is None:
        columns = range(len(clust_idx))
    return pd.DataFrame(_break_down_clusters(c_data.values, clust_idx), index=c_data.index, columns=columns)


def reclustering(data, clust_idx, index=None):
    if index is None:
        index = data.index
        data = data.values
    data = _bulk_up_clusters(data, clust_idx)
    return pd.DataFrame(data, index=index, columns=range(data.shape[1]))


def ss_df_norm(df):
    data = df.values.astype(np.float64)
    data[:] = np.apply_along_axis(_library_size_normalizer, axis=0, arr=data)
    data = zscore(data, axis=1, ddof=DDOF_default)
    return pd.DataFrame(data, index=df.index, columns=df.columns)


def ss_normalization(data, logfunc=DEFAULT_log_transform):
    data[:] = np.apply_along_axis(_library_size_normalizer, axis=0, arr=data)
    data += 1
    data[:] = logfunc(data)
    data[:] = np.apply_along_axis(_interval_normalizer, axis=1, arr=data)
    return True


def _find_optimal_cluster_cut(dist):
    links = linkage(dist, method=DEFAULT_method)
    utils.Debug.vprint("Hierarchial clustering complete: Linkage map constructed")

    ctree = cut_tree(links)
    utils.Debug.vprint("Hierarchial clustering complete: Cut tree constructed")

    start_pos = int(DEFAULT_start_position * ctree.shape[0])
    max_group_size = DEFAULT_max_group_size
    while max_group_size < 0.5:
        for i in list(range(start_pos))[::-1]:
            cslice = ctree[:, i]
            try:
                if _isvalid_cut(cslice, max_group_size=max_group_size):
                    return cslice
            except TooManyClusters:
                break
        max_group_size = max_group_size * 1.1
    raise NoSuitableCutError


def _bulk_up_clusters(data, clust_idx):
    m, n = data.shape
    assert len(clust_idx) == n

    bulk = np.zeros((m, _num_clusters(clust_idx)))
    used_cluster = np.zeros(_num_clusters(clust_idx), dtype=np.dtype(bool))

    for i in range(n):
        used_cluster[clust_idx[i]] = True
        bulk[:, clust_idx[i]] += data[:, i]
    return bulk[:, used_cluster]


def _break_down_clusters(data, clust_idx):
    m, n = data.shape
    assert _num_clusters(clust_idx) == n

    unbulk = np.zeros((m, len(clust_idx)))
    for i in range(len(clust_idx)):
        unbulk[:, i] = data[:, clust_idx[i]]
    return unbulk


def _isvalid_cut(cslice, max_cluster_ratio=DEFAULT_max_cluster_ratio, max_group_size=DEFAULT_max_group_size):
    c_size = cslice.shape[0]

    if _num_clusters(cslice) > c_size * max_cluster_ratio:
        raise TooManyClusters(_num_clusters(cslice))
    if _max_cluster_size(cslice) > c_size * max_group_size:
        return False

    utils.Debug.vprint("Valid cut: {nc} clusters with {ma} maximum size".format(nc=_num_clusters(cslice),
                                                                                ma=_max_cluster_size(cslice)))
    return True


def _pearson_correlation_matrix(data):
    """
    Calculate a pearson correlation matrix for columns
    :param data:
    :return:
    """
    mod_mat = np.var(data, axis=1, ddof=DDOF_default)
    utils.make_array_2d(mod_mat)
    mod_mat = np.sqrt(np.dot(mod_mat, mod_mat.T))
    mod_mat[mod_mat == 0] = 1
    return np.cov(data, ddof=DDOF_default) / mod_mat


def _num_clusters(clust_idx):
    return np.amax(clust_idx) + 1


def _max_cluster_size(clust_idx):
    return np.amax(np.bincount(clust_idx, minlength=len(clust_idx)))


def _library_size_normalizer(vect):
    return vect / np.sum(vect)


def _interval_normalizer(vect):
    if np.max(vect) == np.min(vect):
        return vect
    return (vect - np.min(vect)) / (np.max(vect) - np.min(vect))


class NoSuitableCutError(ArithmeticError):
    pass


class TooManyClusters(ValueError):
    pass
