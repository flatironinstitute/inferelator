import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from . import utils

DDOF_default = 1
DEFAULT_method = 'ward'
DEFAULT_log_transform = np.log10
DEFAULT_start_position = 1

DEFAULT_max_cluster_ratio = 0.25
DEFAULT_start_max_group_size = 0.01
DEFAULT_give_up_max_group_size = 0.25
DEFAULT_step_max_group_size = 1.1


def initial_clustering(data, max_cluster_ratio=DEFAULT_max_cluster_ratio, max_group_size=DEFAULT_give_up_max_group_size,
                       group_size_threshold=DEFAULT_start_max_group_size, group_step=DEFAULT_step_max_group_size):
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
    dist = data.values.astype(np.float64)

    # Normalize the data for library size (per cell) and interval (per gene)
    ss_normalization(dist)
    utils.Debug.vprint("Interval and Library Size Normalization Complete [{}]".format(data.shape))

    # Calculate the distance matrix (1 - Pearson Correlation Coefficient)
    dist = np.corrcoef(dist.T)
    dist *= -1
    dist += 1
    utils.Debug.vprint("Distance matrix construction complete [{}]".format(dist.shape))

    # Convert the distance matrix to a squareform vector
    dist = squareform(dist, force='tovector', checks=False)

    # Perform clustering and find the optimal cluster cut using the default parameters above
    return _find_optimal_cluster_cut(dist, max_cluster_ratio=max_cluster_ratio, max_group_size=max_group_size,
                                     group_size_threshold=group_size_threshold, group_step=group_step)


def make_singles_from_clusters(c_data, clust_idx, columns=None):
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


def make_clusters_from_singles(data, clust_idx, index=None, pseudocount=False):
    if index is None:
        index = data.index
        data = data.values
    data = _bulk_up_clusters(data, clust_idx)

    #Normalize counts so that each cluster is the same library size
    avg_count_per_cluster = np.mean(np.sum(data, axis=0))
    data[:] = np.apply_along_axis(_library_size_normalizer, axis=0, arr=data)

    if pseudocount:
        data *= avg_count_per_cluster

    return pd.DataFrame(data, index=index, columns=range(data.shape[1]))


def ss_df_norm(df):
    data = df.values.astype(np.float64)
    data[:] = np.apply_along_axis(_library_size_normalizer, axis=0, arr=data)
    data = zscore(data, axis=1, ddof=DDOF_default)
    return pd.DataFrame(data, index=df.index, columns=df.columns)


def ss_normalization(data, logfunc=DEFAULT_log_transform):
    # Normalize to library size
    data[:] = np.apply_along_axis(_library_size_normalizer, axis=0, arr=data)

    # Log10 transform x+1 data
    data += 1
    data[:] = logfunc(data)

    # Interval normalize each gene expression (0 to 1)
    data[:] = np.apply_along_axis(_interval_normalizer, axis=1, arr=data)
    return True


def _find_optimal_cluster_cut(dist, max_cluster_ratio, max_group_size, group_size_threshold, group_step):
    links = linkage(dist, method=DEFAULT_method)
    utils.Debug.vprint("Hierarchial clustering complete: Linkage map constructed")

    ctree = cut_tree(links)
    utils.Debug.vprint("Hierarchial clustering complete: Cut tree constructed")

    start_pos = int(DEFAULT_start_position * ctree.shape[0])
    while group_size_threshold < max_group_size:
        for i in list(range(start_pos))[::-1]:
            cslice = ctree[:, i]
            try:
                if _isvalid_cut(cslice, max_cluster_ratio, group_size_threshold):
                    utils.Debug.vprint("{nc} clusters, {ma} maximum".format(nc=_num_clusters(cslice),
                                                                            ma=_max_cluster_size(cslice)))
                    return cslice
            except TooManyClusters:
                break
        group_size_threshold = group_size_threshold * group_step
    raise NoSuitableCutError


def _bulk_up_clusters(data, clust_idx):
    m, n = data.shape
    assert len(clust_idx) == n

    bulk = np.zeros((m, _num_clusters(clust_idx)), dtype='float64')
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


def _isvalid_cut(cslice, max_cluster_ratio, max_group_size):
    if _num_clusters(cslice) > cslice.shape[0] * max_cluster_ratio:
        raise TooManyClusters(_num_clusters(cslice))
    if _max_cluster_size(cslice) > cslice.shape[0] * max_group_size:
        return False
    return True


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
