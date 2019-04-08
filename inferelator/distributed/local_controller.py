"""
LocalController just runs everything in a single process
"""

import collections

from inferelator.distributed import AbstractController
from inferelator.utils import Validator as check


class LocalController(AbstractController):

    _controller_name = "local"

    client = None
    is_master = True
    chunk = None

    @classmethod
    def connect(cls, *args, **kwargs):
        return True

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        return True

    @classmethod
    def map(cls, func, *arg, **kwargs):
        """
        Map a function across iterable(s) and return a list of results

        :param func: function
            Mappable function
        :param args: iterable
            Iterator(s)
        """
        assert check.argument_callable(func)
        assert check.argument_list_type(arg, collections.Iterable)
        return list(map(func, *arg))

    @classmethod
    def shutdown(cls):
        return True
