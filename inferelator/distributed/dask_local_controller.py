from abc import abstractclassmethod
from dask import distributed
import joblib
import tempfile

from inferelator.distributed import AbstractController


class DaskAbstract(AbstractController):
    """
    The DaskController class launches a local dask cluster and connects as a client

    The map functionality is deliberately not implemented; dask-specific multiprocessing functions are used instead
    """

    _controller_name = "dask"
    _controller_dask = True

    _require_initialization = True
    _require_shutdown = True

    client = None

    ## Dask controller variables ##

    # The dask cluster object
    local_cluster = None

    # Settings for dask workers
    processes = 4

    _batch_size = None

    @abstractclassmethod
    def connect(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def shutdown(cls):
        pass

    @classmethod
    def map(
        cls,
        func,
        *args,
        scatter=None,
        restart_workers=False,
        **kwargs
    ):
        """
        Map a function through dask workers

        :param func: Mappable function that takes args and kwargs
        :type func: callable
        :param scatter: Scatter this data to all workers, optional
        :type scatter: list, None
        :param restart_Workers: Restart workers when job is complete
        :type restart_workers: bool
        :raises RuntimeError: Raise a  RuntimeError if the cluster workers arent up
        :return: List of results from the function
        :rtype: list
        """

        if cls.client is None:
            raise RuntimeError(
                "Dask client not initalized, "
                "call .connect() before calling .map()"
            )

        if scatter is not None:
            scatter = {
                id(a): cls.client.scatter([a], hash=False, broadcast=True)[0]
                for a in scatter
            }

            distributed.wait(scatter.values(), timeout=120)

            # Replace kwargs with scattered object
            # If object ID was in the scatter table
            kwargs = {
                k: scatter[id(v)]
                if id(v) in scatter.keys()
                else v
                for k, v in kwargs.items()
            }

        # Quick & dirty batch size heuristic
        if cls._batch_size is None:
            arglen = max(_safe_len(x) for x in args)
            batch_size = int(arglen / cls.num_workers() / 2)
            batch_size = max(2, min(100, batch_size))
        else:
            batch_size = cls._batch_size

        with joblib.parallel_backend(
            'dask',
            client=cls.client
        ):

            res = [
                r for r in joblib.Parallel(
                    batch_size=batch_size
                )(joblib.delayed(func)(
                    *_scatter_wrapper_args(*a, scatter_map=scatter),
                    **kwargs
                )
                for a in zip(*args)
                )
            ]

        if scatter is not None:
            cls.client.cancel(scatter.values())

        if restart_workers:
            cls.client.restart()

        return res

    @abstractclassmethod
    def set_processes(cls, process_count):
        pass

    @classmethod
    def check_cluster_state(cls):
        """
        Is the cluster in a good state to do computation.
        """
        return cls.client is not None

    @classmethod
    def status(cls):
        return cls.check_cluster_state()

    @classmethod
    def num_workers(cls):
        return len(cls.local_cluster.observed)

def _scatter_wrapper_args(*args, scatter_map=None):
    """
    Replace args with dask delayed scatter objects if IDs
    match a lookup table
    """

    if scatter_map is None:
        return args

    else:
        return [
            scatter_map[id(a)]
            if id(a) in scatter_map.keys()
            else a
            for a in args
        ]

def _safe_len(x):
    """
    Length check that's generator-safe
    """

    try:
        return len(x)
    except TypeError:
        return 0


class DaskController(DaskAbstract):

    _controller_name = "dask-local"
    local_dir = None

    _tempdir = None

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        if cls.client is None:

            kwargs["n_workers"] = kwargs.pop("n_workers", cls.processes)
            kwargs["threads_per_worker"] = kwargs.pop("threads_per_worker", 1)
            kwargs["processes"] = kwargs.pop("processes", True)

            # Ugly hack because dask-jobqueue changed this keyword arg
            local_directory = kwargs.pop("local_dir", None)
            local_directory = kwargs.pop("local_directory", None) if local_directory is None else local_directory
            local_directory = cls.local_dir if local_directory is None else local_directory

            if local_directory is None:
                cls._tempdir = tempfile.TemporaryDirectory()
                local_directory = cls._tempdir.name

            kwargs["local_directory"] = local_directory

            cls.local_cluster = distributed.LocalCluster(*args, **kwargs)
            cls.client = distributed.Client(cls.local_cluster)

        return True

    @classmethod
    def set_processes(cls, process_count):
        """
        Set the number of dask workers to use
        :param process_count: int
        :return:
        """
        if cls.client is not None:
            raise RuntimeError(
                "Cannot change worker count on the fly, "
                "shutdown with .shutdown(), set processes with "
                ".set_processes(), and restart with .connnect()"
            )

        cls.processes = process_count

    @classmethod
    def shutdown(cls):

        if cls.client is not None:
            cls.client.close()
            cls.local_cluster.close()

        cls.client = None
        cls.local_cluster = None

        if cls._tempdir is not None:
            cls._tempdir.cleanup()
            cls._tempdir = None

        return True
