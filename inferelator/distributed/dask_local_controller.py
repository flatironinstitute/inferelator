from dask import distributed
import tempfile

from inferelator.distributed.dask import DaskAbstract


class DaskController(DaskAbstract):

    _controller_name = "dask-local"
    local_dir = None

    _tempdir = None

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        if cls.client is None:

            kwargs["n_workers"] = kwargs.pop("n_workers", cls.processes)
            kwargs["threads_per_worker"] = kwargs.pop("threads_per_worker", 1)
            kwargs["processes"] = kwargs.pop("processes", True)

            # Ugly hack because dask-jobqueue changed this keyword arg
            local_directory = kwargs.pop("local_dir", None)
            local_directory = kwargs.pop("local_directory", None) if local_directory is None else local_directory
            local_directory = cls.local_dir if local_directory is None else local_directory

            if local_directory is None:
                cls._tempdir = tempfile.TemporaryDirectory()
                local_directory = cls._tempdir.name

            kwargs["local_directory"] = local_directory

            cls.local_cluster = distributed.LocalCluster(*args, **kwargs)
            cls.client = distributed.Client(cls.local_cluster)

        return True

    @classmethod
    def set_processes(cls, process_count):
        """
        Set the number of dask workers to use
        :param process_count: int
        :return:
        """
        if cls.client is not None:
            raise RuntimeError(
                "Cannot change worker count on the fly, "
                "shutdown with .shutdown(), set processes with "
                ".set_processes(), and restart with .connnect()"
            )

        cls.processes = process_count

    @classmethod
    def shutdown(cls):

        if cls.client is not None:
            cls.client.close()
            cls.local_cluster.close()

        cls.client = None
        cls.local_cluster = None

        if cls._tempdir is not None:
            cls._tempdir.cleanup()
            cls._tempdir = None

        return True
