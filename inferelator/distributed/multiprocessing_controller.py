"""
MultiprocessingController runs everything through a multiprocessing Pool
This requires pathos because the default multiprocessing serializes with cPickle
"""

import pathos
import collections

from inferelator.distributed import AbstractController
from inferelator.utils import Validator as check


class MultiprocessingController(AbstractController):
    _controller_name = "multiprocessing"
    client = None
    is_master = True

    # Control variables
    chunk = 25

    # Num processes
    processes = 4

    @classmethod
    def connect(cls, *args, **kwargs):
        cls.client = pathos.multiprocessing.ProcessPool(processes=cls.processes, **kwargs)
        return True

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        return True

    @classmethod
    def map(cls, func, *args, **kwargs):
        """
        Map a function across iterable(s) and return a list of results

        :param func: function
            Mappable function
        :param args: iterable
            Iterator(s)
        """
        assert check.argument_callable(func)
        assert check.argument_list_type(args, collections.Iterable)
        return cls.client.map(func, *args, chunksize=cls.chunk)

    @classmethod
    def shutdown(cls):
        return cls.client.close()
