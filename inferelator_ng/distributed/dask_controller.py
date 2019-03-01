from inferelator_ng.distributed import AbstractController
from dask import distributed


class DaskController(AbstractController):
    client = None
    is_master = True

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        cls.client = distributed.Client(*args, **kwargs)
        return True

    @classmethod
    def get(cls, *args, **kwargs):
        """
        Dispatch get requests to the dask client
        """

        return cls.client.get(*args, **kwargs)

    @classmethod
    def sync_processes(self, *args, **kwargs):
        """
        This is a thing for KVS. Just return True.
        """

        return True
