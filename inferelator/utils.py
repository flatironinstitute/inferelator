from __future__ import print_function, unicode_literals, division

import pandas as pd
import os

from inferelator.default import SBATCH_VARS

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


class Debug:
    """
    This class is for printing status messages to stdout
    Just plain print doesn't work so well when there are multiple processes
    """
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
        elif lvl in cls.levels.keys():
            cls.verbose_level = cls.levels[lvl]

    @classmethod
    def vprint(cls, *args, **kwargs):
        if cls.silence_clients and not cls.is_master:
            return
        cls.print_level(*args, **kwargs)

    @classmethod
    def allprint(cls, *args, **kwargs):
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
    def argument_numeric(arg, low=None, high=None, allow_none=False, types=(int, float)):
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
    def argument_string(arg, allow_none=False):
        return Validator.argument_type(arg, basestring, allow_none=allow_none)

    @staticmethod
    def argument_list_type(arg, arg_type, allow_none=False):
        if allow_none and arg is None:
            return True
        for a in arg:
            Validator.argument_type(a, arg_type, allow_none=allow_none)
        return True

    @staticmethod
    def argument_callable(arg, allow_none=False):
        if allow_none and arg is None:
            return True
        elif callable(arg):
            return True
        else:
            raise ValueError("Argument {arg} must be callable".format(arg=arg))

    @staticmethod
    def dataframes_align(frame_iterable, allow_none=False, check_order=True):

        is_none = [f is None for f in frame_iterable]
        if any(is_none) and allow_none:
            # If None is an allowed value, remove the Nones and check the remaining dataframes
            new_frame_iterable = []
            for frame in frame_iterable:
                if frame is not None:
                    new_frame_iterable.append(frame)

            # If there are any non-None dataframes, check them for alignment. Otherwise return True
            if len(new_frame_iterable) > 0:
                frame_iterable = new_frame_iterable
            else:
                return True
        elif any(is_none):
            # If None isn't allowed, throw an error
            raise ValueError("None values are present in dataframe list")

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
    def dataframe_is_numeric(frame, allow_none=False):
        if allow_none and frame is None:
            return True

        is_num = frame.applymap(lambda x: isinstance(x, (float, int))).sum()
        is_feature_num = is_num.apply(lambda x: x == frame.shape[0])

        if is_feature_num.all():
            return True
        else:
            bad_features = "\t".join(map(str, is_feature_num.index[is_feature_num].tolist()))
            raise ValueError("Dataframe has non-numeric features: {f}".format(f=bad_features))

    @staticmethod
    def indexes_align(index_iterable, allow_none=False, check_order=True):
        is_none = [f is None for f in index_iterable]
        if any(is_none) and allow_none:
            # If None is an allowed value, remove the Nones and check the remaining dataframes
            new_index_iterable = []
            for index in index_iterable:
                if index is not None:
                    new_index_iterable.append(index)

            # If there are any non-None dataframes, check them for alignment. Otherwise return True
            if len(new_index_iterable) > 0:
                index_iterable = new_index_iterable
            else:
                return True
        elif any(is_none):
            # If None isn't allowed, throw an error
            raise ValueError("None values are present in dataframe list")

        order_flag = False
        zindex = index_iterable[0]
        for ind in index_iterable:
            if len(zindex.difference(ind)) > 0:
                raise ValueError("Indexes have mismatching labels: "+"\t".join(map(str, zindex.difference(ind))))
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
