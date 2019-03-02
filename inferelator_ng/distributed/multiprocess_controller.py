"""
MultiprocessingController runs everything through a multiprocessing Pool
This requires pathos because the default multiprocessing serializes with cPickle
"""

import pathos.multiprocessing as multiprocessing
import collections

from inferelator_ng.distributed import AbstractController
from inferelator_ng.utils import Validator as check


class MultiprocessingController(AbstractController):
    client = None
    is_master = True

    # Control variables
    chunk = 25

    # Num processes
    processes = 4

    @classmethod
    def connect(cls, *args, **kwargs):
        cls.client = multiprocessing.Pool(processes=cls.processes, **kwargs)
        return True

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        return True

    @classmethod
    def map(cls, func, arg, chunk=None, **kwargs):
        assert check.argument_callable(func)
        assert check.argument_type(arg, collections.Iterable)
        assert check.argument_integer(chunk, low=1, allow_none=True)
        chunk = chunk if chunk is not None else cls.chunk
        return list(cls.client.imap(func, arg, chunksize=chunk))
