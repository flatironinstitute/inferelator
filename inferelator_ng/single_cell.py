import pandas as pd
import numpy as np
import itertools

from inferelator_ng.default import DEFAULT_METADATA_FOR_BATCH_CORRECTION
from inferelator_ng.default import DEFAULT_RANDOM_SEED
from inferelator_ng import utils

"""
This file is all preprocessing functions. All functions must take positional arguments expression_matrix and meta_data.
All other arguments must be keyword. All functions must return expression_matrix and meta_data (modified or unmodified).

Normalization functions must take batch_factor_column as a kwarg
Imputation functions must take random_seed as a kwarg
"""


def normalize_expression_to_one(expression_matrix, meta_data,
                                batch_factor_column=DEFAULT_METADATA_FOR_BATCH_CORRECTION):
    """

    :param expression_matrix:
    :param meta_data:
    :param batch_factor_column:
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    utils.Debug.vprint('Normalizing UMI counts per cell ... ')

    # Get UMI counts for each cell
    umi = expression_matrix.sum(axis=1)

    # Divide each cell's raw count data by the total number of UMI counts for that cell
    return expression_matrix.astype(float).divide(umi, axis=0), meta_data


def normalize_medians_for_batch(expression_matrix, meta_data,
                                batch_factor_column=DEFAULT_METADATA_FOR_BATCH_CORRECTION):
    """
    Calculate the median UMI count per cell for each batch. Transform all batches by dividing by a size correction
    factor, so that all batches have the same median UMI count (which is the median batch median UMI count)
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    utils.Debug.vprint('Normalizing median counts between batches ... ')

    # Get UMI counts for each cell
    umi = expression_matrix.sum(axis=1)

    # Create a new dataframe with the UMI counts and the factor to batch correct on
    median_umi = pd.DataFrame({'umi': umi, batch_factor_column: meta_data[batch_factor_column]})

    # Group and take the median UMI count for each batch
    median_umi = median_umi.groupby(batch_factor_column).agg('median')

    # Convert to a correction factor
    median_umi['umi'] = median_umi['umi'] / median_umi['umi'].median()

    # Apply the correction factor to all the data batch-wise. Do this with numpy because pandas is a glacier.
    normed_expression = np.ndarray((0, expression_matrix.shape[1]), dtype=np.dtype(float))
    normed_meta = pd.DataFrame(columns=meta_data.columns)
    for batch, corr_factor in median_umi.iterrows():
        rows = meta_data[batch_factor_column] == batch
        normed_expression = np.vstack((normed_expression, expression_matrix.loc[rows, :].values / corr_factor['umi']))
        normed_meta = pd.concat([normed_meta, meta_data.loc[rows, :]])

    return pd.DataFrame(normed_expression, index=normed_meta.index, columns=expression_matrix.columns), normed_meta


def normalize_multiBatchNorm(expression_matrix, meta_data, batch_factor_column=DEFAULT_METADATA_FOR_BATCH_CORRECTION,
                             minimum_mean=1):
    """
    Normalize as multiBatchNorm from the R package scran
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :param minimum_mean: int
        Minimum mean expression of a gene when considering if it should be included in the correction factor calc
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    utils.Debug.vprint('Normalizing by multiBatchNorm ... ')

    # Calculate size-corrected average gene expression for each batch
    size_corrected_avg = pd.DataFrame(columns=expression_matrix.columns)
    for batch in meta_data[batch_factor_column].unique().tolist():
        batch_df = expression_matrix.loc[meta_data[batch_factor_column] == batch, :]

        # Get UMI counts for each cell
        umi = batch_df.sum(axis=1)
        size_correction_factor = umi / umi.mean()

        # Get the mean size-corrected count values for this batch
        batch_df = batch_df.divide(size_correction_factor, axis=0).mean(axis=0).to_frame().transpose()
        batch_df.index = pd.Index([batch])

        # Append to the dataframe
        size_corrected_avg = size_corrected_avg.append(batch_df)

    # Calculate median ratios
    inter_batch_coefficients = []
    for b1, b2 in itertools.combinations_with_replacement(size_corrected_avg.index.tolist(), r=2):
        # Get the mean size-corrected count values for this batch pair
        b1_series, b2_series = size_corrected_avg.loc[b1, :], size_corrected_avg.loc[b2, :]
        b1_sum, b2_sum = b1_series.sum(), b2_series.sum()

        # calcAverage
        combined_keep_index = ((b1_series / b1_sum + b2_series / b2_sum) / 2 * (b1_sum + b2_sum) / 2) > minimum_mean
        coeff = (b2_series.loc[combined_keep_index] / b1_series.loc[combined_keep_index]).median()

        # Keep track of the median ratios
        inter_batch_coefficients.append((b1, b2, coeff))
        inter_batch_coefficients.append((b2, b1, 1 / coeff))

    inter_batch_coefficients = pd.DataFrame(inter_batch_coefficients, columns=["batch1", "batch2", "coeff"])
    inter_batch_minimum = inter_batch_coefficients.loc[inter_batch_coefficients["coeff"].idxmin(), :]

    min_batch = inter_batch_minimum["batch2"]

    # Apply the correction factor to all the data batch-wise. Do this with numpy because pandas is a glacier.
    normed_expression = np.ndarray((0, expression_matrix.shape[1]), dtype=np.dtype(float))
    normed_meta = pd.DataFrame(columns=meta_data.columns)

    for i, row in inter_batch_coefficients.loc[inter_batch_coefficients["batch2"] == min_batch, :].iterrows():
        select_rows = meta_data[batch_factor_column] == row["batch1"]
        umi = expression_matrix.loc[select_rows, :].sum(axis=1)
        size_correction_factor = umi / umi.mean() / row["coeff"]
        corrected_df = expression_matrix.loc[select_rows, :].divide(size_correction_factor, axis=0).values
        normed_expression = np.vstack((normed_expression, corrected_df))
        normed_meta = pd.concat([normed_meta, meta_data.loc[select_rows, :]])

    return pd.DataFrame(normed_expression, index=normed_meta.index, columns=expression_matrix.columns), normed_meta

def impute_magic_expression(expression_matrix, meta_data, random_seed=DEFAULT_RANDOM_SEED):
    utils.Debug.vprint('Imputing data with MAGIC ... ')
    import magic
    return magic.MAGIC(random_state=random_seed).fit_transform(expression_matrix), meta_data

def impute_SIMLR_expression(expression_matrix, meta_data, random_seed=DEFAULT_RANDOM_SEED):
    # Use MAGIC (van Dijk et al Cell, 2018, 10.1016/j.cell.2018.05.061) to impute data
    utils.Debug.vprint('Imputing data with SIMLR ... ')
    raise NotImplementedError
    import SIMLR
    expression_matrix = SIMLR.helper.fast_pca(expression_matrix, 500)
    expression_matrix, _, _, _ = SIMLR.SIMLR_LARGE(c, 30, 0).fit(expression_matrix)
    return expression_matrix, meta_data

def impute_on_batches(expression_matrix, meta_data, impute_method=impute_magic_expression,
                      random_seed=DEFAULT_RANDOM_SEED, batch_factor_column=DEFAULT_METADATA_FOR_BATCH_CORRECTION):
    """
    Run imputation on separate batches
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :param impute_method: func
        An imputation function from inferelator_ng.single_cell
    :param random_seed: int
        Random seed to put into the imputation method
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    batches = meta_data[batch_factor_column].unique().tolist()
    bc_expression = np.ndarray((0, expression_matrix.shape[1]), dtype=np.dtype(float))
    bc_meta = pd.DataFrame(columns=meta_data.columns)
    for batch in batches:
        rows = meta_data[batch_factor_column] == batch
        batch_corrected, _ = impute_method(expression_matrix.loc[rows, :], None, random_seed=random_seed)
        bc_expression = np.vstack((bc_expression, batch_corrected))
        bc_meta = pd.concat([bc_meta, meta_data.loc[rows, :]])
        random_seed += 1
    return pd.DataFrame(bc_expression, index=bc_meta.index, columns=expression_matrix.columns), bc_meta

def log10_data(expression_matrix, meta_data):
    utils.Debug.vprint('Logging data ... ')
    return np.log10(expression_matrix + 1), meta_data

def log2_data(expression_matrix, meta_data):
    utils.Debug.vprint('Logging data ... ')
    return np.log2(expression_matrix + 1), meta_data

def ln_data(expression_matrix, meta_data):
    utils.Debug.vprint('Logging data ... ')
    return np.log1p(expression_matrix), meta_data

def filter_genes_for_count(expression_matrix, meta_data, count_minimum=None):
    if count_minimum is None:
        return expression_matrix, meta_data
    else:
        keep_genes = expression_matrix.sum(axis=0) >= (count_minimum * expression_matrix.shape[0])
        return expression_matrix.loc[:, keep_genes], meta_data