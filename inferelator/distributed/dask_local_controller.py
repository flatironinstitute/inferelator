from inferelator import default
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

    is_master = True
    client = None

    ## Dask controller variables ##

    # The dask cluster object
    local_cluster = None

    # Settings for dask workers
    processes = default.DEFAULT_PROCESS_COUNT
    local_dir = DEFAULT_LOCAL_DIR

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        kwargs["n_workers"] = kwargs.pop("n_workers", cls.processes)
        kwargs["threads_per_worker"] = kwargs.pop("threads_per_worker", 1)
        kwargs["processes"] = kwargs.pop("processes", True)
        kwargs["local_dir"] = kwargs.pop("local_dir", cls.local_dir)

        cls.local_cluster = distributed.LocalCluster(*args, **kwargs)
        cls.client = distributed.Client(cls.local_cluster)
        return True

    @classmethod
    def shutdown(cls):
        cls.client.close()
        cls.local_cluster.close()

    @classmethod
    def map(cls, func, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def sync_processes(self, *args, **kwargs):
        """
        This is a thing for KVS. Just return True.
        """
        return True
