from __future__ import print_function

import pandas as pd
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
    is_master = True

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
        if cls.silence_clients and not cls.is_master:
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


class Validator(object):
    """
    Validation module for function arguments. Each function here should return True or it should raise an exception
    """

    @staticmethod
    def argument_numeric(arg, low=None, high=None, allow_none=False, types=(int,float)):
        """
        Validate an input argument as being numeric (either an int or a float). Also check bounds if set.
        :param arg:
            Argument to validate
        :param low: numeric
            Lowest (inclusive) acceptable value of the argument; ignore this if it's None
        :param high: numeric
            Lowest (inclusive) acceptable value of the argument; ignore this if it's None
        :param allow_none: bool
            Allow arg to be None if true
        :return:
            Returns True if valid. Raises an exception otherwise
        """
        if allow_none and arg is None:
            return True

        if not isinstance(arg, types):
            raise ValueError("Argument must be numeric ({arg}, {typ} provided) ".format(arg=arg, typ=type(arg)))

        if low is not None and Validator.argument_numeric(low) and arg < low:
            raise ValueError("Argument must be at least {low}".format(low=low))
        if high is not None and Validator.argument_numeric(high) and arg > high:
            raise ValueError("Argument must be no more than {high}".format(high=high))

        return True

    @staticmethod
    def argument_integer(arg, low=None, high=None, allow_none=False):
        """
        Wrapper for argument_numeric which forces only integers
        """
        return Validator.argument_numeric(arg, low=low, high=high, allow_none=allow_none, types=int)

    @staticmethod
    def argument_enum(arg, enum_list, allow_none=False):
        """
        Validate an input argument as being present in an list of acceptable values
        :param arg:
            Argument to validate. If arg is a list or tuple, validate that each element is acceptable
        :param enum_list:
            A list or tuple (or anything that you can use 'in' with; like an index) of valid arguments
        :param allow_none: bool
            Allow arg to be None if true
        :return:
            Returns True if valid. Raises an exception otherwise
        """

        if allow_none and arg is None:
            return True

        if isinstance(arg, (list, tuple)):
            for a in arg:
                Validator.argument_enum(a, enum_list, allow_none=allow_none)
            return True
        elif arg not in enum_list:
            raise ValueError("Argument {arg} must be one of: {enum}".format(arg=arg, enum=",".join(enum_list)))
        else:
            return True

    @staticmethod
    def argument_path(arg, allow_none=False, create_if_needed=False, access=None):
        """
        Check to see if a path exists
        :param arg: str
            Path to a target
        :param allow_none: bool
            Allow arg to be None
        :param create_if_needed: bool
            Create a folder with os.makedirs if the path doesn't exist
        :param access:
            Mode parameter to check access
        :return:
            Returns True if valid. Raises an exception otherwise
        """
        if allow_none and arg is None:
            return True

        # If the path doesn't exist, create it or raise ValueError
        if not os.path.exists(arg) and create_if_needed:
            try:
                os.makedirs(arg)
            except OSError as err:
                raise ValueError("Path {arg} does not exist and cant be created:\n{err}".format(arg=arg, err=str(err)))
        elif not os.path.exists(arg):
            raise ValueError("Argument {arg} must be an existing path".format(arg=arg))

        # If access is set, check and see if the permissions are OK and raise ValueError if not
        if access is not None:
            if os.access(arg, access):
                return True
            else:
                raise ValueError("Path {arg} does not have permission {per}".format(arg=arg, per=access))
        else:
            return True


    @staticmethod
    def argument_type(arg, arg_type, allow_none=False):
        if allow_none and arg is None:
            return True

        if isinstance(arg, arg_type):
            return True
        else:
            raise ValueError("Argument {arg} must be of type {typ}".format(arg=arg, typ=arg_type))

    @staticmethod
    def dataframes_align(frame_iterable, allow_none=False, check_order=True):
        if allow_none and any([f is None for f in frame_iterable]):
            return True

        try:
            Validator.indexes_align([f.index for f in frame_iterable], allow_none=allow_none, check_order=check_order)
        except ValueError as ve:
            raise ValueError("Dataframes are not aligned on indexes: {err}".format(err=str(ve)))

        try:
            Validator.indexes_align([f.columns for f in frame_iterable], allow_none=allow_none, check_order=check_order)
        except ValueError as ve:
            raise ValueError("Dataframes are not aligned on columns: {err}".format(err=str(ve)))

        return True

    @staticmethod
    def indexes_align(index_iterable, allow_none=False, check_order=True):
        if allow_none and any([i is None for i in index_iterable]):
            return True

        order_flag = False
        zindex = index_iterable[0]
        for ind in index_iterable:
            if len(zindex.difference(ind)) > 0:
                raise ValueError("Indexes have mismatching labels")
            elif check_order and any(zindex != ind):
                order_flag = True

        if order_flag:
            raise ValueError("Indexes have matching labels but mismatching order")

        return True

    @staticmethod
    def index_values_unique(index, allow_none=False):
        """
        Check and make sure a pandas index doesn't have duplicate entries
        :param index:
        :param allow_none:
        :return:
        """
        if allow_none and index is None:
            return True
        elif index is None:
            raise ValueError("None is not an acceptable argument")
        elif index.duplicated().sum() > 0:
            dupes = index[index.duplicated()].tolist()
            raise ValueError("Duplicate value(s) present in index: {dupes}".format(dupes=" ".join(dupes)))
        else:
            return True


    @staticmethod
    def arguments_not_none(args, num_none=None):
        """
        :param args:
            Tuple of arguments to check
        :param num_none: int
            The number of arguments which should not be None (so 1 means exactly 1 argument should be not None)
            If None, all arguments should not be None
        :return:
        """
        n_not_none = 0
        for ar in args:
            n_not_none += 0 if ar is None else 1

        if num_none is None and n_not_none != len(args):
            raise ValueError("One of these arguments is None; None is not an acceptable argument")
        elif num_none is not None and n_not_none != num_none:
            raise ValueError("{num} arguments are not None; only {nnum} are allowed".format(num=n_not_none,
                                                                                            nnum=num_none))
        return True

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


def make_array_2d(arr):
    """
    Changes array shape from 1d to 2d if needed (in-place)
    :param arr:  np.ndarray
    """
    if arr.ndim == 1:
        arr.shape = (arr.shape[0], 1)
