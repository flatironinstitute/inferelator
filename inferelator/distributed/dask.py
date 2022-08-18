from abc import abstractmethod
import joblib
from dask import distributed

from inferelator.distributed import AbstractController


class DaskAbstract(AbstractController):
    """
    The DaskAbstract class launches implements the cluster
    mapping function
    It should be extended by a class to build a cluster and
    client for processing
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
    _num_retries = 2

    @classmethod
    @abstractmethod
    def connect(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def shutdown(cls):
        pass

    @classmethod
    def map(
        cls,
        func,
        *args,
        scatter=None,
        restart_workers=False,
        batch_size=None,
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
        :param batch_size: Individual job batch size, optional
        :type batch_size: numeric, None
        :raises RuntimeError: Raise a  RuntimeError if the cluster
            workers arent up
        :return: List of results from the function
        :rtype: list
        """

        if cls.client is None:
            raise RuntimeError(
                "Dask client not initalized, "
                "call .connect() before calling .map()"
            )

        # Scatter things
        # And build a dict of object id to future
        if scatter is not None:
            scatter = {
                id(a): cls.client.scatter(
                    [a],
                    hash=False,
                    broadcast=True
                )[0]
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

        # Quick & dirty batch size heuristic if not passed
        if cls._batch_size is None and batch_size is None:
            arglen = max(_safe_len(x) for x in args)
            batch_size = int(arglen / cls.num_workers() / 2)
            batch_size = max(2, min(100, batch_size))
        elif batch_size is None:
            batch_size = cls._batch_size

        with joblib.parallel_backend(
            'dask',
            client=cls.client,
            retries=cls._num_retries
        ):

            res = [
                r for r in joblib.Parallel(
                    batch_size=batch_size
                )(
                    joblib.delayed(func)(
                        *_scatter_wrapper_args(
                            *a,
                            scatter_map=scatter
                        ),
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

    @classmethod
    @abstractmethod
    def set_processes(cls, process_count):
        pass

    @classmethod
    def set_batch_size(cls, batch_size):
        cls._batch_size = batch_size

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
