import numpy as np
import math
from scipy import sparse as _sparse

from inferelator.utils import Debug, InferelatorData


def make_data_noise(data, random_seed=42):
    """
    Generate a new data object of random data which matches the provided data

    :param data: Raw read data
    :type data: InferelatorData
    :return: Simulated data
    :rtype: InferelatorData
    """

    # Calculate probability vector for gene expression
    # Discrete sampling for count data

    sample_counts = data.sample_counts

    # Normalize to mean counts per sample and sum counts per gene by matrix multiplication
    p_vec = (1 / sample_counts.mean(axis=1)).reshape(1, -1) @ data.expression_data

    if data._is_integer:

        # Flatten and convert counts to a probability vector
        p_vec = p_vec.flatten()
        p_vec = p_vec / p_vec.sum()

        data.expression_data = _sim_ints(p_vec, data.num_obs, sample_counts, sparse=data.is_sparse,
                                         random_seed=random_seed)

    else:

        # Flatten and convert counts to a mean vector
        p_vec = p_vec.flatten()
        p_vec /= data.num_obs

        data.expression_data = _sim_float(p_vec, data.gene_stdev, data.num_obs, random_seed=random_seed)


def _sim_ints(prob_dist, nrows, n_per_row, sparse=False, random_seed=42, sparse_chunk=25000):

    if not np.isclose(np.sum(prob_dist), 1.):
        raise ValueError("Probability distribution does not sum to 1")

    ncols = len(prob_dist)
    rng = np.random.default_rng(seed=random_seed)

    col_ids = np.arange(ncols)

    # Simulate data in a big block
    if not sparse or 2 * sparse_chunk <= nrows:
        synthetic_data = np.zeros((nrows, ncols), dtype=np.uint32)
        for i, u in enumerate(n_per_row):
            synthetic_data[i, :] = np.bincount(rng.choice(col_ids, size=u, p=prob_dist), minlength=ncols)

        synthetic_data = _sparse.csr_matrix(synthetic_data) if sparse else synthetic_data

    # Simulate data in a chunks, make them sparse, and then hstack them
    else:
        synthetic_data = []

        # Make
        for i in range(math.ceil(nrows / sparse_chunk)):
            _nrow_chunk = min(sparse_chunk, nrows - i * sparse_chunk)

            synthetic_chunk = np.zeros((_nrow_chunk, ncols), dtype=np.uint32)
            for j in range(_nrow_chunk):
                _n = n_per_row[i * sparse_chunk + j]
                synthetic_chunk[j, :] = np.bincount(rng.choice(col_ids, size=_n,  p=prob_dist), minlength=ncols)

            synthetic_data.append(_sparse.csr_matrix(synthetic_chunk))
            del synthetic_chunk

        synthetic_data = _sparse.hstack(synthetic_data)

    return synthetic_data


def _sim_float(gene_centers, gene_sds, nrows, random_seed=42):

    ncols = len(gene_centers)
    assert ncols == len(gene_sds)

    rng = np.random.default_rng(seed=random_seed)

    synthetic_data = np.zeros((nrows, ncols), dtype=float)
    for i, (cen, sd) in enumerate(zip(gene_centers, gene_sds)):
        synthetic_data[:, i] = rng.normal(loc=cen, scale=sd, size=nrows)

        synthetic_data = _sparse.csr_matrix(synthetic_data) if sparse else synthetic_data

    return synthetic_data
