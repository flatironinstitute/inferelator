"""
JoblibControler runs everything through joblib
"""

import joblib
import collections.abc

from inferelator.distributed import AbstractController
from inferelator.utils import Validator as check


class JoblibController(AbstractController):
    _controller_name = "joblib"
    _require_initialization = False

    # Num processes
    processes = -1

    @classmethod
    def connect(cls, *args, **kwargs):
        return True

    @classmethod
    def set_processes(cls, process_count):
        """
        Set the number of dask workers to use
        :param process_count: int
        :return:
        """
        check.argument_integer(process_count)

        cls.processes = process_count

    @classmethod
    def map(
        cls,
        func,
        *args,
        scatter=None,
        **kwargs
    ):
        """
        Map a function across iterable(s) and return a list of results

        :param func: Mappable function
        :type func: callable
        :param args: Iterator(s) to pass arguments to func from
        :type args: iterators
        :param scatter: Ignored parameter to match dask requirements
        :type scatter: None
        """

        check.argument_callable(func)
        check.argument_list_type(args, collections.abc.Iterable)

        return [r for r in joblib.Parallel(n_jobs=cls.processes)(
            joblib.delayed(func)(*a, **kwargs) for a in zip(*args)
        )]

    @classmethod
    def shutdown(cls):
        return True
