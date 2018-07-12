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


def convert_to_R_df(df):
    """
    Convert pandas dataframes so that they will be read correctly from CSV format by R.
    """
    new_df = pd.DataFrame(df)
    # Convert booleans to "TRUE" and "FALSE" so they will be read correctly
    for col in new_df.select_dtypes(include=[bool]):
        new_df[col] = [str(x).upper() for x in new_df[col]]
    # Replace null entries with NA entries
    new_df.replace(r'\s+', 'NA', regex=True)
    new_df.replace(np.nan, 'NA', regex=True)
    return new_df


def call_R(driver_path):
    """
    Run an "R" script in a subprocess.
    Any outputs of the script, including stderr, should be saved to files.
    """
    r_command = ["R", "-f", driver_path]
    try:
        subprocess.check_output(r_command, stderr=subprocess.STDOUT, universal_newlines=True)
        return
    except subprocess.CalledProcessError as err:
        print("R command {} failed to execute [Exit code {}]".format(err.cmd, err.returncode))
        print(err.output)
        raise



    # if os.name == "posix":
    #     command = "R -f " + driver_path
    #     return subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    # else:
    #     theproc = subprocess.Popen(['R', '-f', driver_path])
    #     return theproc.communicate()


def r_path(path):
    """
    Convert path to use conventions suitable for use in n "R" script.
    """
    return path.replace('\\', '/')

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

class RDriver:
    """
    Superclass for R driver objects.
    """
    
    target_directory = "/tmp"

    def path(self, filename):
        result = os.path.join(self.target_directory, filename).replace('\\', '/')
        return r_path(result)


def local_path(*location):
    "Return location relative to the folder containing this module."
    return r_path(os.path.join(my_dir, *location))


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
