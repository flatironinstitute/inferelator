import time

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.regression import base_regression
from inferelator import utils

import numpy as np
from dask import distributed

"""
This package contains the dask-specific multiprocessing functions (these are used in place of map calls to allow the
more advanced memory and task tools of dask to be used)
"""

DASK_SCATTER_TIMEOUT = 120


def amusr_regress_dask(X, Y, priors, prior_weight, n_tasks, genes, tfs, G, remove_autoregulation=True, is_restart=False):
    """
    Execute multitask (AMUSR)

    :return: list
        Returns a list of regression results that the amusr_regression pileup_data can process
    """

    assert MPControl.is_dask()

    from inferelator.regression.amusr_regression import format_prior, run_regression_EBIC
    DaskController = MPControl.client

    # Gets genes, n_tasks, prior_weight, and remove_autoregulation from regress_dask()
    # Other arguments are passed in
    def regression_maker(j, x_df, y_list, prior, tf):
        level = 0 if j % 100 == 0 else 2
        utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=genes[j], i=j, total=G),
                             level=level)

        gene = genes[j]
        x, y, tasks = [], [], []

        if remove_autoregulation:
            tf = [t for t in tf if t != gene]
        else:
            pass

        for k, y_data in y_list:
            x.append(x_df[k].loc[:, tf].values)  # list([N, K])
            y.append(y_data)
            tasks.append(k)  # [T,]

        del y_list
        prior = format_prior(prior, gene, tasks, prior_weight)
        return j, run_regression_EBIC(x, y, tf, tasks, gene, prior)

    def response_maker(y_df, i):
        y = []
        gene = genes[i]
        for k in range(n_tasks):
            if gene in y_df[k]:
                y.append((k, y_df[k].loc[:, gene].values.reshape(-1, 1)))
        return y

    # Scatter common data to workers
    [scatter_x] = DaskController.client.scatter([X], broadcast=True)
    [scatter_priors] = DaskController.client.scatter([priors], broadcast=True)

    # Wait for scattering to finish before creating futures
    try:
        distributed.wait(scatter_x, timeout=DASK_SCATTER_TIMEOUT)
        distributed.wait(scatter_priors, timeout=DASK_SCATTER_TIMEOUT)
    except distributed.TimeoutError:
        utils.Debug.vprint("Scattering timeout during regression. Dask workers may be sick", level=0)

    future_list = [DaskController.client.submit(regression_maker, i, scatter_x, response_maker(Y, i), scatter_priors,
                                                tfs)
                   for i in range(G)]

    # Collect results as they finish instead of waiting for all workers to be done
    try:
        result_list = process_futures_into_list(future_list)
    except KeyError:
        utils.Debug.vprint("Unrecoverable job error; restarting")
        if not is_restart:
            return amusr_regress_dask(X, Y, priors, prior_weight, n_tasks, genes, tfs, G,
                                      remove_autoregulation=remove_autoregulation, is_restart=True)
        else:
            raise

    DaskController.client.cancel(scatter_x)
    DaskController.client.cancel(scatter_priors)

    return result_list


def bbsr_regress_dask(X, Y, pp_mat, weights_mat, G, genes, nS, is_restart=False):
    """
    Execute regression (BBSR)

    :return: list
        Returns a list of regression results that the pileup_data can process
    """
    assert MPControl.is_dask()

    from inferelator.regression import bayes_stats
    DaskController = MPControl.client

    def regression_maker(j, x, y, pp, weights):
        level = 0 if j % 100 == 0 else 2
        utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=genes[j], i=j, total=G), level=level)
        data = bayes_stats.bbsr(x, y, pp[j, :].flatten(), weights[j, :].flatten(), nS)
        data['ind'] = j
        return j, data

    # Scatter common data to workers
    [scatter_x] = DaskController.client.scatter([X.values], broadcast=True)
    [scatter_pp] = DaskController.client.scatter([pp_mat.values], broadcast=True)
    [scatter_weights] = DaskController.client.scatter([weights_mat.values], broadcast=True)

    # Wait for scattering to finish before creating futures
    try:
        distributed.wait(scatter_x, timeout=DASK_SCATTER_TIMEOUT)
        distributed.wait(scatter_pp, timeout=DASK_SCATTER_TIMEOUT)
        distributed.wait(scatter_weights, timeout=DASK_SCATTER_TIMEOUT)
    except distributed.TimeoutError:
        utils.Debug.vprint("Scattering timeout during regression. Dask workers may be sick", level=0)

    future_list = [DaskController.client.submit(regression_maker, i, scatter_x, Y.values[i, :].flatten(), scatter_pp,
                                                scatter_weights)
                   for i in range(G)]

    # Collect results as they finish instead of waiting for all workers to be done
    try:
        result_list = process_futures_into_list(future_list)
    except KeyError:
        utils.Debug.vprint("Unrecoverable job error; restarting")
        if not is_restart:
            return bbsr_regress_dask(X, Y, pp_mat, weights_mat, G, genes, nS, is_restart=True)
        else:
            raise

    DaskController.client.cancel(scatter_x)
    DaskController.client.cancel(scatter_pp)
    DaskController.client.cancel(scatter_weights)

    return result_list


def elasticnet_regress_dask(X, Y, params, G, genes, is_restart=False):
    """
    Execute regression (ElasticNet)

    :return: list
        Returns a list of regression results that the pileup_data can process
    """
    assert MPControl.is_dask()

    from inferelator.regression import elasticnet_python
    DaskController = MPControl.client

    def regression_maker(j, x, y):
        level = 0 if j % 100 == 0 else 2
        utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=genes[j], i=j, total=G), level=level)
        data = elasticnet_python.elastic_net(x, y, params=params)
        data['ind'] = j
        return j, data

    # Scatter common data to workers
    [scatter_x] = DaskController.client.scatter([X.values], broadcast=True)

    # Wait for scattering to finish before creating futures
    try:
        distributed.wait(scatter_x, timeout=DASK_SCATTER_TIMEOUT)
    except distributed.TimeoutError:
        utils.Debug.vprint("Scattering timeout during regression. Dask workers may be sick", level=0)

    future_list = [DaskController.client.submit(regression_maker, i, scatter_x, Y.values[i, :].flatten())
                   for i in range(G)]

    # Collect results as they finish instead of waiting for all workers to be done
    try:
        result_list = process_futures_into_list(future_list)
    except KeyError:
        utils.Debug.vprint("Unrecoverable job error; restarting")
        if not is_restart:
            return elasticnet_regress_dask(X, Y, params, G, genes, is_restart=True)
        else:
            raise

    DaskController.client.cancel(scatter_x)

    return result_list


def build_mi_array_dask(X, Y, bins, logtype, is_restart=False):
    """
    Calculate MI into an array with dask (the naive map is very inefficient)

    :param X: np.ndarray (n x m1)
        Discrete array of bins
    :param Y: np.ndarray (n x m2)
        Discrete array of bins
    :param bins: int
        The total number of bins that were used to make the arrays discrete
    :param logtype: np.log func
        Which log function to use (log2 gives bits, ln gives nats)
    :return mi: np.ndarray (m1 x m2)
        Returns the mutual information array
    """

    assert MPControl.is_dask()

    from inferelator.regression.mi import _calc_mi, _make_table

    # Get a reference to the Dask controller
    DaskController = MPControl.client

    m1, m2 = X.shape[1], Y.shape[1]

    def mi_make(i, x, y):
        return i, [_calc_mi(_make_table(x, y[:, j], bins), logtype=logtype) for j in range(m2)]

    # Scatter Y to workers and keep track as Futures
    [scatter_y] = DaskController.client.scatter([Y], broadcast=True, hash=False)
    # Wait for scattering to finish before creating futures
    try:
        distributed.wait(scatter_y, timeout=DASK_SCATTER_TIMEOUT)
    except distributed.TimeoutError:
        utils.Debug.vprint("Scattering timeout during mutual information. Dask workers may be sick", level=0)

    # Build an asynchronous list of Futures for each calculation of mi_make
    future_list = [DaskController.client.submit(mi_make, i, X[:, i], scatter_y, pure=False) for i in range(m1)]

    # Collect results as they finish instead of waiting for all workers to be done
    try:
        mi_list = process_futures_into_list(future_list)
    except KeyError:
        utils.Debug.vprint("Unrecoverable job error; restarting")
        if not is_restart:
            return build_mi_array_dask(X, Y, bins, logtype, is_restart=True)
        else:
            raise

    # Convert the list of lists to an array
    mi = np.array(mi_list)
    assert (m1, m2) == mi.shape, "Array {sh} produced [({m1}, {m2}) expected]".format(sh=mi.shape, m1=m1, m2=m2)

    return mi


def process_futures_into_list(future_list):
    """
    Take a list of futures and turn them into a list of results
    Results must be of the form i, data (where i is the output order)
    :param future_list: list(Futures)
    :return output_list: list(Data)
    """

    DaskController = MPControl.client
    output_list = [None] * len(future_list)
    complete_gen = distributed.as_completed(future_list)

    for finished_future in complete_gen:

        # Jobs can be cancelled in certain situations
        if finished_future.cancelled():
            # Restart cancelled futures and put them back into the work pile
            DaskController.client.retry(finished_future)
            complete_gen.update([finished_future])

        # More likely is jobs erroring as a result of cluster instability
        elif finished_future.status == "error":
            error = finished_future.exception()
            utils.Debug.vprint("Restarting job (Error: {er})".format(er=error), level=1)
            # Restart errored futures and put them back into the work pile
            DaskController.client.retry(finished_future)
            complete_gen.update([finished_future])

        # In the event of success, get the data
        i, result_data = finished_future.result()
        output_list[i] = result_data

    return output_list
