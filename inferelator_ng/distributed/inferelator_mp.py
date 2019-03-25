from inferelator_ng.distributed import AbstractController
from inferelator_ng import utils
from inferelator_ng import default

# Python 2/3 compatible string checking
try:
    basestring
except NameError:
    basestring = str


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
    def set_multiprocess_engine(cls, engine):
        """
        Register the multiprocessing engine to use

        Currently available are:

        dask-cluster
        dask-local
        kvs
        multprocessing
        local

        :param engine: str / Controller object
            A string to lookup the controller or a Controller object
        """

        if isinstance(engine, basestring):
            if engine == "dask-cluster":
                from inferelator_ng.distributed.dask_cluster_controller import DaskSLURMController
                cls.client = DaskSLURMController
            elif engine == "dask-local":
                from inferelator_ng.distributed.dask_local_controller import DaskController
                cls.client = DaskController
            elif engine == "kvs":
                from inferelator_ng.distributed.kvs_controller import KVSController
                cls.client = KVSController
            elif engine == "multiprocessing":
                from inferelator_ng.distributed.multiprocessing_controller import MultiprocessingController
                cls.client = MultiprocessingController
            elif engine == "local":
                from inferelator_ng.distributed.local_controller import LocalController
                cls.client = LocalController
            else:
                raise ValueError("Engine {eng_str} unknown".format(eng_str=engine))
        elif issubclass(engine, AbstractController):
            cls.client = engine
        else:
            raise ValueError("Engine must be provided as a string for lookup or an implemented Controller class object")

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Connect to the manager or scheduler or process pool or whatever using the `.connect()` implementation in the
        multiprocessing engine.
        """
        if cls.is_initialized:
            return True

        if cls.client is None:
            cls.set_multiprocess_engine(default.DEFAULT_MULTIPROCESSING_ENGINE)

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
            raise ConnectionError("Connect before calling map()")
        return cls.client.map(*args, **kwargs)

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        """
        Make sure processes are all at the same point by calling the `.sync_processes()` implementation in the
        multiprocessing engine

        This is necessary for KVS; other engines will just return True
        """
        if not cls.is_initialized:
            raise ConnectionError("Connect before calling sync_processes()")
        return cls.client.sync_processes(*args, **kwargs)

    @classmethod
    def shutdown(cls):
        """
        Gracefully shut down the multiprocessing engine by calling `.shutdown()`
        """

        if cls.is_initialized:
            client = cls.client.shutdown()
            cls.is_initialized = False
        else:
            client = True

        return client
