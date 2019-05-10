# This is the required package init. Everything in this package must implement the abstract class AbstractController

from abc import abstractmethod


class AbstractController:
    # The object which handles the multiprocessing
    client = None

    # Boolean to identify master processes where needed
    is_master = False

    # The chunk sizes for calls to map
    chunk = 25

    # The name of this controller
    _controller_name = None
    _controller_dask = False

    @classmethod
    def name(cls):
        """
        This returns the _class_name which all subclasses should define
        """
        if cls._controller_name is None:
            raise NameError("Controller name has not been defined")
        return cls._controller_name

    @classmethod
    def is_dask(cls):
        return cls._controller_dask

    @classmethod
    @abstractmethod
    def connect(cls, *args, **kwargs):
        """
        This establishes or creates a multiprocessing state that allows `map` or other functionality to be used
        in parallel
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def map(cls, *args, **kwargs):
        """
        This implements a map function that will execute in parallel. If this is not implemented, the standard
        workflow will fail and provision must be made elsewhere to define the multiprocessing functionality
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sync_processes(cls, *args, **kwargs):
        """
        This synchronizes multiple processes. Multiprocessing methods which have a defined hierarchy and no risk of
        race conditions may simply return True
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def shutdown(cls):
        """
        Clean shutdown of the multiprocessing state
        """
        raise NotImplementedError
