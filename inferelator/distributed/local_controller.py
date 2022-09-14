"""
LocalController just runs everything in a single process
"""

import collections.abc

from inferelator.distributed import AbstractController
from inferelator import utils
from inferelator.utils import Validator as check


class LocalController(AbstractController):

    _controller_name = "local"

    _require_initialization = False
    _require_shutdown = False

    client = None
    chunk = None

    @classmethod
    def connect(cls, *args, **kwargs):
        return True

    @classmethod
    def map(
        cls,
        func,
        *arg,
        scatter=None,
        **kwargs
    ):
        """
        Map a function across iterable(s) and return a list of results

        :param func: function
            Mappable function
        :param args: iterable
            Iterator(s)
        """
        assert check.argument_callable(func)
        assert check.argument_list_type(arg, collections.abc.Iterable)
        return list(map(lambda *x: func(*x, **kwargs), *arg))

    @classmethod
    def set_processes(cls, process_count):
        """
        Set the number of dask workers to use
        :param process_count: int
        :return:
        """
        utils.Debug.vprint("Local does not support multiple cores", level=0)

    @classmethod
    def shutdown(cls):
        return True
