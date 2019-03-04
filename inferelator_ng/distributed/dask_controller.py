import collections

from inferelator_ng.distributed import AbstractController
from inferelator_ng.utils import Validator as check

from dask import distributed
from toolz import partition_all


class DaskController(AbstractController):
    client = None
    chunk = 25
    is_master = True
    processes = 4

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        kwargs["n_workers"] = kwargs.pop("n_workers", cls.processes)
        kwargs["threads_per_worker"] = 1
        kwargs["processes"] = True

        cls.client = distributed.Client(distributed.LocalCluster(*args, **kwargs))
        return True

    @classmethod
    def map(cls, func, iterable, chunk=None, **kwargs):
        """
        Map a function across an iterable and return a list of results
        :param func: function
            Mappable function
        :param iterable: iterable
            Iterator
        :param chunk: int
            The number of iterations to assign in blocks
        :return:
        """

        assert check.argument_callable(func)
        assert check.argument_type(iterable, collections.Iterable)
        assert check.argument_integer(chunk, low=1, allow_none=True)
        chunk = chunk if chunk is not None else cls.chunk

        # Function that returns a list of mapped results for a chunk of data
        def chunker(block):
            return [func(individual) for individual in block]

        # Build Futures and then gather them into a nested list of results
        future_list = cls.client.map(chunker, partition_all(chunk, iterable))
        nested_list = cls.client.gather(future_list)

        # Flatten the list
        flat_list = []
        for chunk_list in nested_list:
            flat_list.extend(chunk_list)

        return flat_list

    @classmethod
    def sync_processes(self, *args, **kwargs):
        """
        This is a thing for KVS. Just return True.
        """

        return True
