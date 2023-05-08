import pandas as pd
import pandas.api.types as pat
import os
import inspect


class Validator:
    """
    Validation module for function arguments. Each function here should return
    True or it should raise an exception
    """

    @staticmethod
    def argument_numeric(
        arg,
        low=None,
        high=None,
        allow_none=False,
        types=(int, float)
    ):
        """
        Validate an input argument as being numeric (either an int or a float).
        Also check bounds if set.

        :param arg:
            Argument to validate
        :param low: numeric
            Lowest (inclusive) acceptable value of the argument;
            ignore this if it's None
        :param high: numeric
            Lowest (inclusive) acceptable value of the argument;
            ignore this if it's None
        :param allow_none: bool
            Allow arg to be None if true
        :return:
            Returns True if valid. Raises an exception otherwise
        """
        if allow_none and arg is None:
            return True

        if not isinstance(arg, types):
            raise ValueError(
                f"Argument must be numeric ({arg}, {type(arg)} provided)"
            )

        if low is not None and arg < low:
            raise ValueError(
                f"Argument must be at least {low}, {arg} provided"
            )
        if high is not None and arg > high:
            raise ValueError(
                f"Argument must be no more than {high}, {arg} provided"
            )

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
    def argument_subpath(arg, is_subpath_of, allow_none=False):
        """
        Check and see if an argument is a subpath of a path
        :param arg: str
            Path
        :param is_subpath_of: str
            Path for comparison
        :param allow_none: bool
            Allow arg to be None
        :return:
        """

        if allow_none and arg is None:
            return True
        elif arg is None or is_subpath_of is None:
            raise ValueError("Path argument {a} cannot be compared to path {b}".format(a=arg, b=is_subpath_of))

        arg = os.path.abspath(os.path.expanduser(arg))
        is_subpath_of = os.path.abspath(os.path.expanduser(is_subpath_of))

        if arg == is_subpath_of:
            return True
        elif is_subpath_of == os.path.abspath(os.sep):
            return True
        elif arg.startswith(is_subpath_of + os.sep):
            return True
        else:
            raise ValueError("Path {a} is not a subpath of path {b}".format(a=arg, b=is_subpath_of))

    @staticmethod
    def argument_type(
        arg,
        arg_type,
        allow_none=False
    ):

        if allow_none and arg is None:
            return True

        if isinstance(arg, arg_type):
            return True
        else:
            raise ValueError(
                f"Argument {arg} must be of type {arg_type} "
                f"({type(arg)} provided)"
            )

    @staticmethod
    def argument_string(arg, allow_none=False):
        return Validator.argument_type(arg, str, allow_none=allow_none)

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

        non_numeric = pd.Index([not pat.is_numeric_dtype(x) for x in frame.dtypes])

        if non_numeric.any():
            bad_features = "\t".join(map(str, frame.columns[non_numeric].tolist()))
            raise ValueError("Dataframe has non-numeric features: {f}".format(f=bad_features))
        else:
            return True

    @staticmethod
    def dataframe_is_finite(frame, allow_none=False, check_index=True):
        if allow_none and frame is None:
            return True

        with pd.option_context('mode.use_inf_as_na', True):
            non_finites = frame.apply(lambda x: pd.isnull(x).sum()) > 0
            if non_finites.any():
                bad_features = "\t".join(map(str, frame.columns[non_finites].tolist()))
                raise ValueError("Dataframe has non-finite features: {f}".format(f=bad_features))
            elif check_index and pd.isnull(frame.index).any():
                raise ValueError("NaN values are present in frame index")
            elif check_index and pd.isnull(frame.columns).any():
                raise ValueError("NaN values are present in frame column")
            else:
                return True

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

    @staticmethod
    def argument_is_subclass(arg, subclass, allow_none=False):
        """

        :param arg:
            Argument to check
        :param subclass:
            Class that the argument must be a subclass of
        :param allow_none:
        """
        if allow_none and arg is None:
            return True
        elif arg is None:
            raise ValueError("None is not an acceptable argument")

        if not inspect.isclass(arg) and inspect.isclass(type(arg)):
            arg = type(arg)

        if not inspect.isclass(subclass):
            raise ValueError("Subclass to test argument is itself not a class")
        elif not issubclass(arg, subclass):
            raise ValueError("Argument is not a subclass of {sc}".format(sc=str(subclass)))
        else:
            return True


def is_string(arg):
    """
    Check if a argument is a string in a python 2/3 compatible way
    :param arg:
    :return:
    """
    return isinstance(arg, str)
