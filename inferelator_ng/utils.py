"""
Miscellaneous utility modules.
"""

from __future__ import print_function
import os
import pandas as pd
import subprocess
import numpy as np

my_dir = os.path.dirname(__file__)


class Debug:
    verbose_level = 0
    default_level = 1

    levels = dict(normal=0,
                  verbose=1, v=1,
                  very_verbose=2, vv=2,
                  max_output=3, vvv=3)

    @classmethod
    def set_verbose_level(cls, lvl):
        if isinstance(lvl, (int, float)):
            cls.verbose_level = lvl

    @classmethod
    def vprint(cls, *args, **kwargs):
        try:
            level = kwargs.pop('level')
        except KeyError:
            level = cls.default_level
        if level <= cls.verbose_level:
            print(*args, **kwargs)
        else:
            pass

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

    #If we're the main process, set KVS count to 0
    if 0 == rank:
        kvs.put('count', 0)

    #Start at the baseline
    checks, lower, upper = 0, -1, -1

    while 1:

        #Checks increments every loop
        #If it's greater than the upper bound, get a new lower bound from the KVS count
        #Set the new upper bound by adding chunk to lower
        #And then put the new upper bound back into KVS count

        if checks >= upper:
            lower = kvs.get('count')
            upper = lower + chunk
            kvs.put('count', upper)

        #Yield TRUE if this row belongs to this process and FALSE if it doesn't
        yield lower <= checks < upper
        checks += 1

def kvsTearDown(kvs, rank):
    # de-initialize the global counter.        
    if 0 == rank:
        # Do a hard reset if rank == 0                                                                                                       
        kvs.get('count')

def df_from_tsv(file_like, has_index = True):
    "Read a tsv file or buffer with headers and row ids into a pandas dataframe."
    return pd.read_csv(file_like, sep="\t", header=0, index_col= 0 if has_index else False)

def metadata_df(file_like):
    "Read a metadata file as a pandas data frame."
    return pd.read_csv(file_like, sep="\t", header=0, index_col="condName")

def read_tf_names(file_like):
    "Read transcription factor names from one-column tsv file.  Return list of names."
    exp = pd.read_csv(file_like, sep="\t", header=None)
    assert exp.shape[1] == 1, "transcription factor file should have one column "
    return list(exp[0])
