from __future__ import print_function
import pandas as pd
import numpy as np
import os

# Get the following environment variables
# Workflow_variable_name, casting function, default (if the env isn't set or the casting fails for whatever reason)
SBATCH_VARS = {'RUNDIR': ('output_dir', str, None),
               'DATADIR': ('input_dir', str, None),
               'SLURM_PROCID': ('rank', int, 0),
               'SLURM_NTASKS_PER_NODE': ('cores', int, 1),
               'SLURM_NTASKS': ('tasks', int, 1),
               'SLURM_NODEID': ('node', int, 0),
               'SLURM_JOB_NUM_NODES': ('num_nodes', int, 1)
               }


def slurm_envs():
    envs = {}
    for os_var, (cv, mt, de) in SBATCH_VARS.items():
        try:
            val = mt(os.environ[os_var])
        except (KeyError, TypeError):
            val = de
        envs[cv] = val
    return envs


class Debug:
    verbose_level = 0
    default_level = 1

    silence_clients = True
    rank = slurm_envs()['rank']

    levels = dict(silent=-1,
                  normal=0,
                  verbose=1, v=1,
                  very_verbose=2, vv=2,
                  max_output=3, vvv=3)

    @classmethod
    def set_verbose_level(cls, lvl):
        if isinstance(lvl, (int, float)):
            cls.verbose_level = lvl

    @classmethod
    def vprint(cls, *args, **kwargs):
        if cls.silence_clients and cls.rank != 0:
            return
        cls.print_level(*args, **kwargs)

    @classmethod
    def warn(cls, *args, **kwargs):
        cls.vprint(*args, level=cls.levels["v"], **kwargs)

    @classmethod
    def notify(cls, *args, **kwargs):
        cls.vprint(*args, level=cls.levels["vv"], **kwargs)

    @classmethod
    def vprint_all(cls, *args, **kwargs):
        cls.print_level(*args, **kwargs)

    @classmethod
    def print_level(cls, *args, **kwargs):
        try:
            level = kwargs.pop('level')
        except KeyError:
            level = cls.default_level
        if level <= cls.verbose_level:
            print((" " * level), *args, **kwargs)
        else:
            return


def df_from_tsv(file_like, has_index=True):
    "Read a tsv file or buffer with headers and row ids into a pandas dataframe."
    return pd.read_csv(file_like, sep="\t", header=0, index_col=0 if has_index else False)


def metadata_df(file_like):
    "Read a metadata file as a pandas data frame."
    return pd.read_csv(file_like, sep="\t", header=0, index_col="condName")


def read_tf_names(file_like):
    "Read transcription factor names from one-column tsv file.  Return list of names."
    exp = pd.read_csv(file_like, sep="\t", header=None)
    assert exp.shape[1] == 1, "transcription factor file should have one column "
    return list(exp[0])


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


def bool_to_index(arr):
    """
    Returns an array that indexes all the True elements of a boolean array
    :param arr: np.ndarray
    :return: np.ndarray
    """
    return np.where(arr)[0]


def index_of_nonzeros(arr):
    """
    Returns an array that indexes all the non-zero elements of an array
    :param arr: np.ndarray
    :return: np.ndarray
    """
    return np.where(arr != 0)[0]


def make_array_2d(arr):
    """
    Changes array shape from 1d to 2d if needed (in-place)
    :param arr:  np.ndarray
    """
    if arr.ndim == 1:
        arr.shape = (arr.shape[0], 1)
