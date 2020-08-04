import pandas as pd
import numpy as np

from inferelator import utils

"""
This file is all preprocessing functions. All functions must take positional arguments data (InferelatorData).
All other arguments must be keyword. All functions must change the data object in place and return nothing.

Normalization functions take batch_factor_column [str] as a kwarg
Imputation functions take random_seed [int] and output_file [str] as a kwarg 
"""


def normalize_expression_to_one(data, **kwargs):
    """

    :param data: InferelatorData [N x G]
    """

    utils.Debug.vprint('Normalizing UMI counts per cell ... ')

    # Divide each cell's raw count data by the total number of UMI counts for that cell
    data.divide(data.sample_counts, axis=1)


def normalize_expression_to_median(data, **kwargs):
    """

    :param data: InferelatorData [N x G]
    """

    target_value = np.median(data.sample_counts)
    data.divide((data.sample_counts / target_value), axis=1)


def normalize_medians_for_batch(data, batch_factor_column=None, **kwargs):
    """
    Calculate the median UMI count per cell for each batch. Transform all batches by dividing by a size correction
    factor, so that all batches have the same median UMI count (which is the median batch median UMI count)

    :param data: InferelatorData [N x G]
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    if batch_factor_column is None or batch_factor_column not in data.meta_data:
        _msg = "batch_factor_column must be set to one of the meta data columns"
        utils.Debug.vprint(_msg + ": {c}".format(c=data.meta_data.columns), level=0)
        raise ValueError(_msg)

    utils.Debug.vprint('Normalizing median counts between batches ... ')

    # Get UMI counts for each cell
    umi = data.sample_counts

    # Create a new dataframe with the UMI counts and the factor to batch correct on
    umi = pd.DataFrame({'umi': umi, batch_factor_column: data.meta_data[batch_factor_column]})

    # Group and take the median UMI count for each batch
    median_umi = umi.groupby(batch_factor_column).agg('median')

    # Convert to a correction factor based on the median of the medians
    median_umi = median_umi / median_umi['umi'].median()
    umi = umi.join(median_umi, on=batch_factor_column, how="left", rsuffix="_mod")

    # Apply the correction factor to all the data
    data.divide(umi['umi_mod'].values, axis=1)


def normalize_sizes_within_batch(data, batch_factor_column=None, **kwargs):
    """
    Calculate the median UMI count within each batch and then resize each sample so that each sample has the same total
    UMI count

    :param data: InferelatorData [N x G]
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    if batch_factor_column is None or batch_factor_column not in data.meta_data:
        _msg = "batch_factor_column must be set to one of the meta data columns"
        utils.Debug.vprint(_msg + ": {c}".format(c=data.meta_data.columns), level=0)
        raise ValueError(_msg)

    utils.Debug.vprint('Normalizing to median counts within batches ... ')

    # Get UMI counts for each cell
    umi = data.sample_counts

    # Create a new dataframe with the UMI counts and the factor to batch correct on
    umi = pd.DataFrame({'umi': umi, batch_factor_column: data.meta_data[batch_factor_column]})

    # Group and take the median UMI count for each batch
    median_umi = umi.groupby(batch_factor_column).agg('median')

    # Convert to a correction factor based on the median of the medians
    umi = umi.join(median_umi, on="Condition", how="left", rsuffix="_mod")
    umi['umi_mod'] = umi['umi'] / umi['umi_mod']

    # Apply the correction factor to all the data
    data.divide(umi['umi_mod'].values, axis=1)


def log10_data(data, **kwargs):
    """
    Transform the expression data by adding one and then taking log10. Ignore any kwargs.

    :param data: InferelatorData [N x G]
    """
    utils.Debug.vprint('Logging data [log10+1] ... ')
    data.transform(np.log10, add_pseudocount=True)


def log2_data(data, **kwargs):
    """
    Transform the expression data by adding one and then taking log2. Ignore any kwargs.

    :param data: InferelatorData [N x G]
    """
    utils.Debug.vprint('Logging data [log2+1]... ')
    data.transform(np.log2, add_pseudocount=True)


def ln_data(data, **kwargs):
    """
    Transform the expression data by adding one and then taking ln. Ignore any kwargs.

    :param data: InferelatorData [N x G]
    """
    utils.Debug.vprint('Logging data [ln+1]... ')
    data.transform(np.log1p, add_pseudocount=False)


def tf_sqrt_data(data, **kwargs):
    """
    Transform the expression data by sqrt(x) + sqrt(x+1) and restore sparsity with x - 1
    Based on Freeman & Tukey: https://projecteuclid.org/euclid.aoms/1177729756

    :param data: InferelatorData [N x G]
    """
    utils.Debug.vprint('Freeman-Tukey square root transformation [sqrt(x) + sqrt(x+1) - 1]... ')
    data.transform(lambda x: np.sqrt(x) + np.sqrt(x + 1) - 1)


def filter_genes_for_count(data, count_minimum=None):
    """
    Filter out any genes which have a variance of 0 by calling filter_genes_for_var. Filter out any genes which don't
    reach the minimum count (if count is not none)

    :param data: InferelatorData [N x G]
    :param count_minimum: num
        The minimum value per sample required to include any genes
   """

    if count_minimum is None:
        data.trim_genes(remove_constant_genes=True)
    else:
        count_minimum = count_minimum * data.shape[0]
        if np.min(data.expression_data.min(axis=0)) < 0:
            raise ValueError("Cannot use a count minimum on data with negative values")
        counts_per_gene = data.gene_counts
        if np.any(~np.isfinite(counts_per_gene)):
            raise ValueError("Non-finite values in count matrix")
        keep_genes = counts_per_gene >= count_minimum
        utils.Debug.vprint("Filtering {gn} genes [Count]".format(gn=data.shape[1] - np.sum(keep_genes)), level=1)
        data.trim_genes(remove_constant_genes=True, trim_gene_list=data.gene_names[keep_genes])
