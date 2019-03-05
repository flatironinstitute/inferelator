import collections

# Maintain python 2 compatibility
try:
    from itertools import izip as zip
except ImportError:
    pass

from inferelator_ng.distributed import AbstractController
from inferelator_ng.utils import Validator as check

from dask import distributed


class DaskController(AbstractController):
    client = None
    chunk = 25
    is_master = True
    processes = 4

    local_cluster = None

    @classmethod
    def name(cls):
        return "dask"

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        kwargs["n_workers"] = kwargs.pop("n_workers", cls.processes)
        kwargs["threads_per_worker"] = 1
        kwargs["processes"] = True

        cls.local_cluster = distributed.LocalCluster(*args, **kwargs)
        cls.client = distributed.Client(cls.local_cluster)
        return True

    @classmethod
    def shutdown(cls):
        cls.client.close()
        cls.local_cluster.close()

    @classmethod
    def map(cls, func, *args, **kwargs):
        """
        Map a function across iterable(s) and return a list of results

        :param func: function
            Mappable function
        :param args: iterable
            Iterator(s)
        :param chunk: int
            The number of iterations to assign in blocks
        :return:
        """

        assert check.argument_callable(func)
        assert check.argument_list_type(args, collections.Iterable)

        # Function that returns a list of mapped results for a chunk of data
        # Each chunk is a list containing zip(*arg) elements with length cls.processes
        def operate_on_chunk(block):
            return [func(*individual) for individual in block]

        # Function that takes a list of arguments and chunks it
        def create_chunk(n, arg_list):
            chunk_data = list(zip(*arg_list))
            for s in range(int(len(chunk_data) / n) + 1):
                yield chunk_data[s * n:min(s * n + n, len(chunk_data))]

        # Build Futures and then gather them into a nested list of results
        # future_list = [delayed(operate_on_chunk)(block) for block in create_chunk(cls.chunk, args)]
        # nested_list = compute(*future_list, scheduler="distributed")

        # Build Futures with map and then gather them into a nested list of results
        future_list = cls.client.map(operate_on_chunk, create_chunk(cls.chunk, args))
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
