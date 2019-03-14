from inferelator_ng.distributed import AbstractController
from inferelator_ng.distributed.kvs_controller import KVSController
from inferelator_ng import utils

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

    _class_name = "multiprocessing_registry"
    client = None

    # Relevant external state booleans
    is_master = False
    is_initialized = False

    @classmethod
    def set_multiprocess_engine(cls, engine):

        if isinstance(engine, basestring):
            if engine == "dask-cluster":
                from inferelator_ng.distributed.dask_cluster_controller import DaskSLURMController
                cls.client = DaskSLURMController
            elif engine == "kvs":
                from inferelator_ng.distributed.kvs_controller import KVSController

        cls.client = engine

    @classmethod
    def connect(cls, *args, **kwargs):
        if cls.is_initialized:
            return True
        connect_return = cls.client.connect(*args, **kwargs)

        # Set the process state
        cls.is_master = cls.client.is_master
        cls.is_initialized = True

        # Also tell Debug if we're the master process
        utils.Debug.is_master = cls.is_master

        return connect_return

    @classmethod
    def map(cls, *args, **kwargs):
        if not cls.is_initialized:
            raise ConnectionError("Connect before calling map()")
        return cls.client.map(*args, **kwargs)

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        if not cls.is_initialized:
            raise ConnectionError("Connect before calling sync_processes()")
        return cls.client.sync_processes(*args, **kwargs)

    @classmethod
    def shutdown(cls):
        return cls.client.shutdown() if cls.is_initialized else True
