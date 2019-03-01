# This is the required package init. Everything in this package must implement the abstract class AbstractController

import collections

from abc import abstractmethod
from inferelator_ng.utils import Validator as check


class AbstractController:
    client = None
    is_master = False

    @classmethod
    @abstractmethod
    def connect(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sync_processes(cls, *args, **kwargs):
        raise NotImplementedError


def process_dask_graph(dsk, result):
    """
    This processes simplistic dask graphs for the kvs or local controller
    :param dsk: dict
    :param result: func, list, list, list
    :return:
    """
    assert check.argument_type(result, collections.Hashable)

    # If result points to a function tuple, start unpacking. Otherwise just return it
    if isinstance(dsk[result], tuple):
        func = dsk[result][0]
    else:
        return dsk[result]

    # If the function tuple is just a function, execute it and then return it
    if len(dsk[result]) == 1:
        return func, None, None, None
    else:
        func_args = dsk[result][1:]

    # Unpack arguments and map anything that's got data in the graph
    map_args = []
    for arg in func_args:
        try:
            map_args.append(dsk[arg])
        except (TypeError, KeyError):
            map_args.append(arg)

    # Find out which arguments should be iterated over
    iter_args = [isinstance(arg, (tuple, list)) for arg in map_args]
    iter_product = []

    # If nothing is iterable, call the function and return it
    if sum(iter_args) == 0:
        return func, map_args, None, None

    # Put the iterables in a list
    for iter_bool, arg in zip(iter_args, map_args):
        if iter_bool:
            iter_product.append(arg)

    return func, map_args, iter_args, iter_product


def process_dask_function_args(map_args, iter_args, iterated_args):
    iter_arg_idx = 0
    current_args = []

    # Pack up this iteration's arguments into a list
    for iter_bool, arg in zip(iter_args, map_args):
        if iter_bool:
            current_args.append(iterated_args[iter_arg_idx])
            iter_arg_idx += 1
        else:
            current_args.append(arg)

    return current_args
