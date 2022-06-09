from inferelator.distributed import AbstractController
from inferelator import utils

DEFAULT_MP_ENGINE = "joblib"


class MPControl(AbstractController):
    """
    This is the multiprocessing controller. It is a pass-through for the method-specific multiprocessing implementations
    A multiprocessing implementation can be registered here and then used throughout the inferelator
    """

    _controller_name = "multiprocessing_registry"
    client = None

    @classmethod
    def name(cls):
        """
        This returns the _class_name which all subclasses should define
        """
        if cls.client is None:
            raise NameError("No client has been set")
        return cls.client.name()

    @classmethod
    def is_dask(cls):
        """
        This returns True if dask functions should be used
        """
        if cls.client is None:
            return False
        return cls.client._controller_dask

    @classmethod
    def set_multiprocess_engine(cls, engine, processes=None):
        """
        Register the multiprocessing engine to use

        Currently available are:

        dask-cluster
        dask-k8
        dask-local
        multiprocessing
        joblib
        local

        :param engine: A string to lookup the controller or a Controller object
        :type engine: str, Controller
        :param processes: Number of processes to use. Equivalent to calling `set_processes`
        :type processes: int
        """

        if utils.is_string(engine):
            engine = engine.lower()

            # If this is already the selected engine, move on
            if cls.client is not None and cls.client.name() == engine:
                pass
            elif cls.client is not None and cls.client.name() == 'joblib' and engine == 'multiprocessing':
                pass

            # Check to see if there's something running that has to be handled
            elif cls.client is not None and cls.status() and cls.client._require_shutdown:
                raise RuntimeError("Client is currently active. Run .shutdown() before changing engines.")

            # Dask engines
            elif engine == "dask-cluster":
                from inferelator.distributed.dask_cluster_controller import DaskHPCClusterController
                cls.client = DaskHPCClusterController
            elif engine == "dask-local":
                from inferelator.distributed.dask_local_controller import DaskController
                cls.client = DaskController
            elif engine == "dask-k8":
                from inferelator.distributed.dask_k8_controller import DaskK8Controller
                cls.client = DaskK8Controller

            # Old, dead key-value storage server engine
            elif engine == "kvs":
                raise DeprecationWarning("The KVS engine is deprecated. Use Dask-based multiprocessing")

            # Local engines
            elif engine == "multiprocessing" or engine == "joblib":
                from inferelator.distributed.joblib_controller import JoblibController
                cls.client = JoblibController
            elif engine == "local":
                from inferelator.distributed.local_controller import LocalController
                cls.client = LocalController
            else:
                raise ValueError(f"Engine {engine} is not a valid argument")

        elif issubclass(engine, AbstractController):
            cls.client = engine

        else:
            raise ValueError("Engine must be provided as a string for lookup or an implemented Controller class object")

        utils.Debug.vprint("Inferelator parallelized using engine {eng}".format(eng=cls.name()))

        if processes is not None:
            cls.set_processes(processes)

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Connect to the manager or scheduler or process pool or whatever using the `.connect()` implementation in the
        multiprocessing engine.
        """

        if cls.client is None:
            utils.Debug.vprint(
                f"Loading default engine {DEFAULT_MP_ENGINE}"
            )

            cls.set_multiprocess_engine(DEFAULT_MP_ENGINE)

        return cls.client.connect(*args, **kwargs)

    @classmethod
    def map(cls, func, *args, **kwargs):
        """
        Map using the `.map()` implementation in the multiprocessing engine
        """

        if cls.client is None:
            cls.connect()

        return cls.client.map(func, *args, **kwargs)

    @classmethod
    def set_processes(cls, process_count):
        """
        Set worker process count
        """

        if cls.client is None:
            cls.connect()

        return cls.client.set_processes(process_count)

    @classmethod
    def shutdown(cls):
        """
        Gracefully shut down the multiprocessing engine by calling `.shutdown()`
        """

        if cls.client is not None:
            return cls.client.shutdown()
        else:
            return None

    @classmethod
    def status(cls):
        """
        True if ready to process data, False otherwise
        """

        if cls.client is None:
            return False
        elif not cls.client._require_initialization:
            return True
        elif hasattr(cls.client, 'client') and cls.client.client is not None:
            return True
        elif hasattr(cls.client, 'client') and cls.client.client is None:
            return False
        else:
            return False
