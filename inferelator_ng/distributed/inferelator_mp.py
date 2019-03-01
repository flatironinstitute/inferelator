from inferelator_ng.distributed.kvs_controller import KVSController
from inferelator_ng import utils

class MPControl:
    """
    This is the multiprocessing controller
    """

    # Which multiprocessing engine to use
    multiprocess_engine = KVSController

    # Relevant external state booleans
    is_master = False
    is_initialized = False

    @classmethod
    def set_multiprocess_engine(cls, engine):
        cls.multiprocess_engine = engine
        return True

    @classmethod
    def connect(cls, *args, **kwargs):
        if cls.is_initialized:
            return True
        connect_return = cls.multiprocess_engine.connect(*args, **kwargs)

        # Set the process state
        cls.is_master = cls.multiprocess_engine.is_master
        cls.is_initialized = True

        # Also tell Debug if we're the master process
        utils.Debug.is_master = cls.is_master

        return connect_return

    @classmethod
    def get(cls, *args, **kwargs):
        return cls.multiprocess_engine.get(*args, **kwargs)

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        return cls.multiprocess_engine.sync_processes(*args, **kwargs)

