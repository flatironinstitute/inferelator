from inferelator_ng.distributed.kvs_controller import KVSController


class MPControl:
    """
    This is the multiprocessing controller
    """
    multiprocess_engine = KVSController
    is_master = False

    @classmethod
    def set_multiprocess_engine(cls, engine):
        cls.multiprocess_engine = engine
        return True

    @classmethod
    def connect(cls, *args, **kwargs):
        connect_return = cls.multiprocess_engine.connect(*args, **kwargs)
        cls.is_master = cls.multiprocess_engine.is_master
        return connect_return

    @classmethod
    def get(cls, *args, **kwargs):
        return cls.multiprocess_engine.get(*args, **kwargs)

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        return cls.multiprocess_engine.sync_processes(*args, **kwargs)

