from __future__ import print_function, unicode_literals, division

import pandas as pd
import os

from inferelator.default import SBATCH_VARS
from inferelator.utils.validator import Validator
from inferelator.utils.debug import Debug
from inferelator.utils.data import InferelatorData

# Python 2/3 compatible string checking
try:
    basestring
except NameError:
    basestring = str


def slurm_envs(var_names=None):
    """
    Get environment variable names and return them as a dict
    :param var_names: list
        A list of environment variable names to get. Will throw an error if they're not keys in the SBATCH_VARS dict
    :return envs: dict
        A dict keyed by setattr variable name of the value (or default) from the environment variables
    """
    var_names = SBATCH_VARS.keys() if var_names is None else var_names
    assert set(var_names).issubset(set(SBATCH_VARS.keys()))

    envs = {}
    for cv in var_names:
        os_var, mt, de = SBATCH_VARS[cv]
        try:
            val = mt(os.environ[os_var])
        except (KeyError, TypeError):
            val = de
        envs[cv] = val
    return envs

def df_from_tsv(file_like, has_index=True):
    "Read a tsv file or buffer with headers and row ids into a pandas dataframe."
    return pd.read_csv(file_like, sep="\t", header=0, index_col=0 if has_index else False)


def df_set_diag(df, val, copy=True):
    """
    Sets the diagonal of a dataframe to a value. Diagonal in this case is anything where row label == column label.

    :param df: pd.DataFrame
        DataFrame to modify
    :param val: numeric
        Value to insert into any cells where row label == column label
    :param copy: bool
        Force-copy the dataframe instead of modifying in place
    :return: pd.DataFrame / int
        Return either the modified dataframe (if copied) or the number of cells modified (if changed in-place)
    """

    # Find all the labels that are shared between rows and columns
    isect = df.index.intersection(df.columns)

    if copy:
        df = df.copy()

    # Set the value where row and column names are the same
    for i in range(len(isect)):
        df.loc[isect[i], isect[i]] = val

    if copy:
        return df
    else:
        return len(isect)


def is_string(arg):
    """
    Check if a argument is a string in a python 2/3 compatible way
    :param arg:
    :return:
    """
    return isinstance(arg, basestring)


def make_array_2d(arr):
    """
    Changes array shape from 1d to 2d if needed (in-place)
    :param arr:  np.ndarray
    """
    if arr.ndim == 1:
        arr.shape = (arr.shape[0], 1)


def melt_and_reindex_dataframe(data_frame, value_name, idx_name="target", col_name="regulator"):
    """
    Take a pandas dataframe and melt it into a one column dataframe (with the column `value_name`) and a multiindex
    of the original index + column
    :param data_frame: pd.DataFrame [M x N]
        Meltable dataframe
    :param value_name: str
        The column name for the values of the dataframe
    :param idx_name: str
        The name to assign to the original data_frame index values
    :param col_name: str
        The name to assign to the original data_frame column values
    :return: pd.DataFrame [(M*N) x 1]
        Melted dataframe with a single column of values and a multiindex that is the original index + column for
        that value
    """

    assert Validator.argument_type(data_frame, pd.DataFrame)

    # Copy the dataframe and move the index to a column
    data_frame = data_frame.copy()
    data_frame[idx_name] = data_frame.index

    # Melt it into a [(M*N) x 3] dataframe
    data_frame = data_frame.melt(id_vars=idx_name, var_name=col_name, value_name=value_name)

    # Create a multiindex and then drop the columns that are now in the index
    data_frame.index = pd.MultiIndex.from_frame(data_frame.loc[:, [idx_name, col_name]])
    del data_frame[idx_name]
    del data_frame[col_name]

    return data_frame
