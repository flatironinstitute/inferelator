# This is the required package init. Everything in this package must implement the abstract class AbstractController

from abc import abstractmethod


class AbstractController:

    # The object which handles the multiprocessing
    client = None

    # Boolean to identify master processes where needed
    is_master = False

    # The chunk sizes for calls to map
    chunk = 25

    @classmethod
    @abstractmethod
    def name(cls):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def connect(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def map(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sync_processes(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def shutdown(cls):
        raise NotImplementedError