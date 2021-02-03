import numpy as np
import math
from scipy import sparse as _sparse

from inferelator.utils import Debug
from inferelator.distributed.inferelator_mp import MPControl


def make_data_noisy(data, random_seed=42):
    """
    Generate a new data object of random data which matches the provided data

    :param data: Raw read data
    :type data: InferelatorData
    :param random_seed: Random seed for data generation
    :type random_seed: int
    :return: Simulated data
    :rtype: InferelatorData
    """

    # Calculate probability vector for gene expression
    # Discrete sampling for count data

    sample_counts = data.sample_counts

    if data._is_integer:

        Debug.vprint("Simulating integer count data for {n} samples".format(n=data.num_obs), level=0)

        # Data is centered already
        if np.any(sample_counts <= 0.):
            p_vec = np.ones(data.num_genes, dtype=float)

        # Normalize to mean counts per sample and sum counts per gene by matrix multiplication
        else:
            p_vec = (np.mean(sample_counts) / sample_counts).reshape(1, -1) @ data.expression_data

        # Flatten and convert counts to a probability vector
        p_vec = p_vec.flatten()
        p_vec = p_vec / p_vec.sum()

        data.expression_data = _sim_ints(p_vec, sample_counts, sparse=data.is_sparse, random_seed=random_seed)

    else:

        # Data is centered already
        if np.any(sample_counts <= 0.):
            p_vec = np.zeros(data.num_genes, dtype=float)

        # Normalize to mean total measured values per sample and sum counts per gene by matrix multiplication
        else:
            p_vec = (np.mean(sample_counts) / sample_counts).reshape(1, -1) @ data.expression_data
            p_vec /= data.num_obs

        Debug.vprint("Simulating float data for {n} samples".format(n=data.num_obs), level=0)
        data.expression_data = _sim_float(p_vec.flatten(), data.gene_stdev, data.num_obs, random_seed=random_seed)


def _sim_ints(prob_dist, n_per_row, sparse=False, random_seed=42):

    if not np.isclose(np.sum(prob_dist), 1.):
        raise ValueError("Probability distribution does not sum to 1")

    ncols = len(prob_dist)

    def _sim_rows(n_vec, seed):
        row_data = np.zeros((len(n_vec), ncols), dtype=np.int32)

        rng = np.random.default_rng(seed=seed)
        col_ids = np.arange(ncols)

        for i, n in enumerate(n_vec):
            row_data[i, :] = np.bincount(rng.choice(col_ids, size=n, p=prob_dist), minlength=ncols)

        return _sparse.csr_matrix(row_data) if sparse else row_data

    ss = np.random.SeedSequence(random_seed)
    sim_data = MPControl.map(_sim_rows, _row_gen(n_per_row), _ss_gen(ss))

    return _sparse.vstack(sim_data) if sparse else np.vstack(sim_data)


def _sim_float(gene_centers, gene_sds, nrows, random_seed=42):

    ncols = len(gene_centers)
    assert ncols == len(gene_sds)

    def _sim_cols(cents, sds, seed):
        rng = np.random.default_rng(seed=seed)
        return rng.normal(loc=cents, scale=sds, size=(nrows, len(cents)))

    ss = np.random.SeedSequence(random_seed)

    return np.hstack(MPControl.map(_sim_cols, _col_gen(gene_centers), _col_gen(gene_sds), _ss_gen(ss)))


def _row_gen(n_vec, chunksize=2000):
    _chunks = math.ceil(len(n_vec) / chunksize)
    for i in range(_chunks):
        yield n_vec[i * chunksize: min(len(n_vec), (i + 1) * chunksize)]


def _col_gen(vals, chunksize=200):
    _chunks = math.ceil(len(vals) / chunksize)
    for i in range(_chunks):
        _start, _stop = i * chunksize, min(len(vals), (i + 1) * chunksize)
        yield vals[_start: _stop]


def _ss_gen(ss):
    while True:
        yield ss.generate_state(1)[0]

