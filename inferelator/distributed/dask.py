from abc import abstractmethod

from dask import distributed
import joblib

from inferelator.distributed import AbstractController
from inferelator.utils import Debug

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

    _batch_size = 1
    _num_retries = 2
    _restart_workers = False

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
        **kwargs
    ):
        """
        Map a function through dask workers

        :param func: Mappable function that takes args and kwargs
        :type func: callable
        :param scatter: Scatter this data to all workers, optional
        :type scatter: list, None
        :raises RuntimeError: Raise a RuntimeError if the cluster
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
            scatter = make_scatter_map(
                scatter,
                cls.client
            )

            # Replace kwargs with scattered object
            # If object ID was in the scatter table
            kwargs = {
                k: scatter[id(v)]
                if id(v) in scatter.keys()
                else v
                for k, v in kwargs.items()
            }

        # Submit directly and process futures if batchsize is 1
        if cls._batch_size == 1:

            futures = [
                cls.client.submit(
                    func,
                    *_scatter_wrapper_args(
                        *a,
                        scatter_map=scatter
                    ),
                    retries=cls._num_retries,
                    **kwargs,
                )
                for a in zip(*args)
            ]

            res = process_futures_into_list(
                futures,
                cls.client
            )

        # Submit through joblib if batchsize > 1
        # Has some weird behavior sometimes
        # Isn't properly checking for some error conditions
        # Can race/end up in infinite wait state
        # I think it's because this is submitting coroutines
        # Instead of sending functions to worker processes
        else:
            with joblib.parallel_backend(
                'dask',
                client=cls.client,
                retries=cls._num_retries
            ):

                res = [
                    r for r in joblib.Parallel(
                        batch_size=cls._batch_size
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

        if cls._restart_workers:
            cls.client.restart()

        return res

    @classmethod
    @abstractmethod
    def set_processes(cls, process_count):
        pass

    @classmethod
    def set_task_parameters(
        cls,
        batch_size=None,
        restart_workers=None,
        retries=None
    ):
        """
        Set parameters for submitted worker tasks

        :param batch_size: Batch into submissions of this size,
            defaults to 1
        :type batch_size: int, optional
        :param restart_workers: Restart workers after every map,
            defaults to False
        :type restart_workers: bool, optional
        :param retries: Number of times to retry failed jobs,
            defaults to 2
        :type retries: int, optional
        """
        cls.set_param("_batch_size", batch_size)
        cls.set_param("_restart_workers", restart_workers)
        cls.set_param("_num_retries", retries)

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

def make_scatter_map(scatter_objs, client):

    """
    Scatter stuff and return a dict of
    object id: dask.Future for that object

    :param scatter_objs: list or iterable of objects to scatter
    :type scatter_objs: list, tuple, iterable
    :param client: Dask client
    :type client: distributed.Client
    :return: Dict {id(obj):distributed.Future(obj)}
    :rtype: dict
    """

    if scatter_objs is None:
        return None

    # Generate a dict, keyed by ID
    # Of dask futures
    scatter = {
        id(a): client.scatter(
            [a],
            hash=False,
            broadcast=True
        )[0]
        for a in scatter_objs
    }

    distributed.wait(scatter.values(), timeout=120)

    return scatter

def process_futures_into_list(
    future_list,
    client,
    raise_on_error=True,
    check_results=True
):
    """
    Take a list of futures and turn them into a list of results

    :param future_list: A list of executing futures
    :type future_list: list
    :param client: A dask client
    :type: distributed.Client
    :param raise_on_error: Should an error be raised if a job can't be
        restarted or just move on from it.
    Defaults to True
    :type raise_on_error: bool
    :param check_results: Should the result object be checked (and
        restarted if there's a problem). If False, this will raise
        an error with the result of a failed future is retrieved.
        Defaults to True.
    :type check_results: bool

    :return output_list: A list of results from the completed futures
    :rtype: list
    """

    output_list = [None] * len(future_list)

    # Make a dict to look up positions
    # To retain final ordering of results
    position_lookup = {
        id(f): i
        for i, f in enumerate(future_list)
    }

    complete_gen = distributed.as_completed(future_list)

    for finished_future in complete_gen:

        # Check possible error states
        _is_error = finished_future.cancelled()
        _is_error |= (finished_future.status == "erred")

        # Jobs can be cancelled in certain situations
        if check_results and _is_error:

            Debug.vprint(
                f"Restarting job (Error: {finished_future.exception()})",
                level=0
            )

            # Restart cancelled futures and put them back into the work pile
            try:
                client.retry(finished_future)
                complete_gen.update([finished_future])
            except KeyError:
                if raise_on_error:
                    raise
                else:
                    Debug.vprint(
                        f"Job {id(finished_future)} failed",
                        level=0
                    )

        # In the event of success, get the data
        else:
            result_data = finished_future.result()
            output_list[position_lookup[id(finished_future)]] = result_data
            finished_future.cancel()

    return output_list
