from inferelator.distributed import AbstractController
from inferelator import utils
import warnings

DEFAULT_MP_ENGINE = "local"


class MPControl(AbstractController):
    """
    This is the multiprocessing controller. It is a pass-through for the method-specific multiprocessing implementations
    A multiprocessing implementation can be registered here and then used throughout the inferelator
    """

    _controller_name = "multiprocessing_registry"
    client = None

    # Relevant external state booleans
    is_master = False
    is_initialized = False

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
        return cls.client.is_dask()

    @classmethod
    def set_multiprocess_engine(cls, engine, processes=None):
        """
        Register the multiprocessing engine to use

        Currently available are:

        dask-cluster
        dask-k8
        dask-local
        kvs
        multiprocessing
        local

        :param engine: A string to lookup the controller or a Controller object
        :type engine: str, Controller
        :param processes: Number of processes to use. Equivalent to calling `set_processes`
        :type processes: int
        """
        if cls.is_initialized:
            raise RuntimeError("Client is currently active. Run .shutdown() before changing engines.")

        if utils.is_string(engine):
            if engine == "dask-cluster":
                from inferelator.distributed.dask_cluster_controller import DaskHPCClusterController
                cls.client = DaskHPCClusterController
            elif engine == "dask-local":
                from inferelator.distributed.dask_local_controller import DaskController
                cls.client = DaskController
            elif engine == "dask-k8":
                from inferelator.distributed.dask_k8_controller import DaskK8Controller
                cls.client = DaskK8Controller
            elif engine == "kvs":
                warnings.warn("The KVS engine is deprecated. It has been replaced by Dask-based multiprocessing",
                              DeprecationWarning)
                from inferelator.distributed.kvs_controller import KVSController
                cls.client = KVSController
            elif engine == "multiprocessing":
                from inferelator.distributed.multiprocessing_controller import MultiprocessingController
                cls.client = MultiprocessingController
            elif engine == "local":
                from inferelator.distributed.local_controller import LocalController
                cls.client = LocalController
            else:
                raise ValueError("Engine {eng_str} unknown".format(eng_str=engine))
        elif issubclass(engine, AbstractController):
            cls.client = engine
        else:
            raise ValueError("Engine must be provided as a string for lookup or an implemented Controller class object")

        utils.Debug.vprint("Inferelator MPControl using engine {eng}".format(eng=cls.name()))

        if processes is not None:
            cls.set_processes(processes)

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Connect to the manager or scheduler or process pool or whatever using the `.connect()` implementation in the
        multiprocessing engine.
        """
        if cls.is_initialized:
            return True

        if cls.client is None:
            utils.Debug.vprint("Loading default engine {eng}".format(eng=DEFAULT_MP_ENGINE))
            cls.set_multiprocess_engine(DEFAULT_MP_ENGINE)

        connect_return = cls.client.connect(*args, **kwargs)

        # Set the process state
        cls.is_master = cls.client.is_master
        cls.is_initialized = True

        # Also tell Debug if we're the master process
        utils.Debug.is_master = cls.is_master

        return connect_return

    @classmethod
    def map(cls, *args, **kwargs):
        """
        Map using the `.map()` implementation in the multiprocessing engine
        """
        if not cls.is_initialized:
            raise RuntimeError("Connect before calling map()")
        return cls.client.map(*args, **kwargs)

    @classmethod
    def set_processes(cls, process_count):
        """
        Set worker process count
        """
        if cls.is_initialized:
            raise RuntimeError("Cannot set processes after the engine has started")
        return cls.client.set_processes(process_count)

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        """
        Make sure processes are all at the same point by calling the `.sync_processes()` implementation in the
        multiprocessing engine

        This is necessary for KVS; other engines will just return True
        """
        if not cls.is_initialized:
            raise RuntimeError("Connect before calling sync_processes()")
        return cls.client.sync_processes(*args, **kwargs)

    @classmethod
    def shutdown(cls):
        """
        Gracefully shut down the multiprocessing engine by calling `.shutdown()`
        """

        if cls.is_initialized:
            client_off = cls.client.shutdown()
            cls.is_initialized = False
            cls.client = None
        else:
            client_off = True

        return client_off
