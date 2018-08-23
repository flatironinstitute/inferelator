from __future__ import print_function
import pandas as pd
import numpy as np
import os

# Get the following environment variables
# Workflow_variable_name, casting function, default (if the env isn't set or the casting fails for whatever reason)
SBATCH_VARS = {'RUNDIR': ('output_dir', str, None),
               'DATADIR': ('input_dir', str, None),
               'SLURM_PROCID': ('rank', int, 0),
               'SLURM_NTASKS_PER_NODE': ('cores', int, 10),
               'SLURM_NTASKS': ('tasks', int, 1)
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

    silence_clients=True
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
        try:
            level = kwargs.pop('level')
        except KeyError:
            level = cls.default_level
        if level <= cls.verbose_level:
            print((" " * level), *args, **kwargs)
        else:
            return

    @classmethod
    def warn(cls, *args, **kwargs):
        cls.vprint(*args, level=cls.levels["v"], **kwargs)

    @classmethod
    def notify(cls, *args, **kwargs):
        cls.vprint(*args, level=cls.levels["vv"], **kwargs)


def ownCheck(kvs, rank, chunk=1):
    """
    Generator

    :param kvs: KVS object
    :param rank: SLURM proc ID
    :param chunk: The size of the chunk given to each subprocess

    :yield: bool
    """
    # initialize a global counter.                                                                                                               

    # If we're the main process, set KVS count to 0
    if 0 == rank:
        kvs.put('count', 0)

    # Start at the baseline
    checks, lower, upper = 0, -1, -1

    while 1:

        # Checks increments every loop
        # If it's greater than the upper bound, get a new lower bound from the KVS count
        # Set the new upper bound by adding chunk to lower
        # And then put the new upper bound back into KVS count

        if checks >= upper:
            lower = kvs.get('count')
            upper = lower + chunk
            kvs.put('count', upper)

        # Yield TRUE if this row belongs to this process and FALSE if it doesn't
        yield lower <= checks < upper
        checks += 1


def kvsTearDown(kvs, rank):
    # de-initialize the global counter.        
    if 0 == rank:
        # Do a hard reset if rank == 0                                                                                                       
        kvs.get('count')

def kvs_sync_processes(kvs, rank, pref=""):
    # Block all processes until they reach this point
    # Then release them
    # It may be wise to use unique prefixes if this is gonna get called rapidly so there's no collision
    # Or not. I'm a comment, not a cop.

    n = slurm_envs()['tasks']
    wkey = pref + '_wait'
    ckey = pref + '_continue'

    kvs.put(wkey, True)
    if rank == 0:
        for _ in range(n):
            kvs.get(wkey)
        for _ in range(n):
            kvs.put(ckey, True)
    kvs.get(ckey)

class FakeKVS:
    """
    A fake KVS client that only does one thing poorly. Only for troubleshooting when this is not in a workflow.
    """
    data = {}

    def put(self, key, val):
        self.data[key] = val

    def get(self, key):
        try:
            return self.data[key]
        except KeyError:
            return None


def always_true():
    while True:
        yield True


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
