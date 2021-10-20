from inferelator.distributed.inferelator_mp import MPControl
from inferelator.regression import base_regression
from inferelator import utils
import copy

import numpy as np
import scipy.sparse as sps
from dask import distributed

"""
This package contains the dask-specific multiprocessing functions (these are used in place of map calls to allow the
more advanced memory and task tools of dask to be used)
"""

DASK_SCATTER_TIMEOUT = 120

def dask_map(func, *args, **kwargs):
    """
    Dask map

    :param func: function to map
    :type func: callable
    :param args: positional arguments for func
    :type args: iterable
    :param kwargs: keyword (non-iterable) arguments for func. Keywords will be passed to dask client.map.
    :return: List of mapped results
    :rtype: list
    """

    assert MPControl.is_dask()

    def _func_caller(f, i, *a, **k):
        return i, f(*a, **k)

    return process_futures_into_list([MPControl.client.client.submit(_func_caller, func, i, *za, **kwargs)
                                      for i, za in enumerate(zip(*args))])


def process_futures_into_list(future_list, raise_on_error=True, check_results=True):
    """
    Take a list of futures and turn them into a list of results
    Results must be of the form i, data (where i is the output order)

    :param future_list: A list of executing futures
    :type future_list: list
    :param raise_on_error: Should an error be raised if a job can't be restarted or just move on from it.
    Defaults to True
    :type raise_on_error: bool
    :param check_results: Should the result object be checked (and restarted if there's a problem)
    If False, this will raise an error with the result of a failed future is retrieved.
    Defaults to True.
    :type check_results: bool
    :return output_list: A list of results from the completed futures
    :rtype: list
    """

    DaskController = MPControl.client
    output_list = [None] * len(future_list)
    complete_gen = distributed.as_completed(future_list)

    for finished_future in complete_gen:

        DaskController.check_cluster_state()

        # Jobs can be cancelled in certain situations
        if check_results and (finished_future.cancelled() or (finished_future.status == "erred")):
            error = finished_future.exception()
            utils.Debug.vprint("Restarting job (Error: {er})".format(er=error), level=0)

            # Restart cancelled futures and put them back into the work pile
            try:
                DaskController.client.retry(finished_future)
                complete_gen.update([finished_future])
            except KeyError:
                if raise_on_error:
                    raise

        # In the event of success, get the data
        else:
            i, result_data = finished_future.result()
            output_list[i] = result_data
            finished_future.cancel()

    return output_list

