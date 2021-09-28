from inferelator.distributed.dask_functions import dask_map
import os

# Maintain python 2 compatibility
try:
    from itertools import izip as zip
except ImportError:
    pass

from inferelator.distributed import AbstractController

from dask import distributed

try:
    DEFAULT_LOCAL_DIR = os.environ['TMPDIR']
except KeyError:
    DEFAULT_LOCAL_DIR = 'dask-worker-space'


class DaskController(AbstractController):
    """
    The DaskController class launches a local dask cluster and connects as a client

    The map functionality is deliberately not implemented; dask-specific multiprocessing functions are used instead
    """

    _controller_name = "dask-local"
    _controller_dask = True

    client = None

    ## Dask controller variables ##

    # The dask cluster object
    local_cluster = None

    # Settings for dask workers
    processes = 4
    local_dir = DEFAULT_LOCAL_DIR

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        kwargs["n_workers"] = kwargs.pop("n_workers", cls.processes)
        kwargs["threads_per_worker"] = kwargs.pop("threads_per_worker", 1)
        kwargs["processes"] = kwargs.pop("processes", True)

        # Ugly hack because dask-jobqueue changed this keyword arg
        local_directory = kwargs.pop("local_dir", None)
        local_directory = kwargs.pop("local_directory", None) if local_directory is None else local_directory
        kwargs["local_directory"] = local_directory if local_directory is not None else cls.local_dir

        cls.local_cluster = distributed.LocalCluster(*args, **kwargs)
        cls.client = distributed.Client(cls.local_cluster)
        return True

    @classmethod
    def shutdown(cls):
        cls.client.close()
        cls.local_cluster.close()

    @classmethod
    def map(cls, func, *args, **kwargs):
        return dask_map(func, *args, **kwargs)

    @classmethod
    def set_processes(cls, process_count):
        """
        Set the number of dask workers to use
        :param process_count: int
        :return:
        """
        cls.processes = process_count

    @classmethod
    def check_cluster_state(cls):
        """
        This is a thing for the non-local dask. Just return True.
        """
        return True

    @classmethod
    def is_dask(cls):
        """
        This is dask. Return True.
        """
        return True