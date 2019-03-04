from inferelator_ng.distributed import AbstractController
from inferelator_ng.distributed.kvs_controller import KVSController
from inferelator_ng import utils


class MPControl(AbstractController):
    """
    This is the multiprocessing controller
    """

    # Which multiprocessing engine to use
    client = KVSController

    # Relevant external state booleans
    is_master = False
    is_initialized = False
    is_dask = False

    @classmethod
    def name(cls):
        return "multiprocessing_registry"

    @classmethod
    def set_multiprocess_engine(cls, engine):
        cls.client = engine
        return True

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
