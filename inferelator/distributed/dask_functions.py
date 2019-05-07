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


def amusr_regress_dask(X, Y, priors, prior_weight, n_tasks, genes, tfs, G, remove_autoregulation=True):
    """
    Execute multitask (AMUSR)

    :return: list
        Returns a list of regression results that the amusr_regression pileup_data can process
    """

    assert MPControl.name() == "dask"

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
    result_list = [None] * len(future_list)
    for finished_future, (j, result_data) in distributed.as_completed(future_list, with_results=True):
        result_list[j] = result_data
        finished_future.cancel()

    DaskController.client.cancel(scatter_x)
    DaskController.client.cancel(scatter_priors)

    return result_list


def bbsr_regress_dask(X, Y, pp_mat, weights_mat, G, genes, nS):
    """
    Execute regression (BBSR)

    :return: list
        Returns a list of regression results that the pileup_data can process
    """
    assert MPControl.name() == "dask"

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
    result_list = [None] * len(future_list)
    for finished_future, (j, result_data) in distributed.as_completed(future_list, with_results=True):
        result_list[j] = result_data
        finished_future.cancel()

    DaskController.client.cancel(scatter_x)
    DaskController.client.cancel(scatter_pp)
    DaskController.client.cancel(scatter_weights)

    return result_list


def elasticnet_regress_dask(X, Y, params, G, genes):
    """
    Execute regression (ElasticNet)

    :return: list
        Returns a list of regression results that the pileup_data can process
    """
    assert MPControl.name() == "dask"

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
    result_list = [None] * len(future_list)
    for finished_future, (j, result_data) in distributed.as_completed(future_list, with_results=True):
        result_list[j] = result_data
        finished_future.cancel()

    DaskController.client.cancel(scatter_x)

    return result_list


def build_mi_array_dask(X, Y, bins, logtype):
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

    assert MPControl.name() == "dask"

    from inferelator.regression.mi import _calc_mi, _make_table

    # Get a reference to the Dask controller
    dask_controller = MPControl.client

    m1, m2 = X.shape[1], Y.shape[1]

    def mi_make(i, x, y):
        level = 1 if i % 1000 == 0 else 3
        utils.Debug.allprint("Mutual Information Calculation [{i} / {total}]".format(i=i, total=m1), level=level)
        return i, [_calc_mi(_make_table(x, y[:, j], bins), logtype=logtype) for j in range(m2)]

    # Scatter Y to workers and keep track as Futures
    try:
        [scatter_y] = dask_controller.client.scatter([Y], broadcast=True)
        # Wait for scattering to finish before creating futures
        try:
            distributed.wait(scatter_y, timeout=DASK_SCATTER_TIMEOUT)
        except distributed.TimeoutError:
            utils.Debug.vprint("Scattering timeout during mutual information. Dask workers may be sick", level=0)
    except AssertionError:
        # There is something wrong with distributed.replicate - failover to non-broadcast and see if it works
        [scatter_y] = dask_controller.client.scatter([Y])

    # Build an asynchronous list of Futures for each calculation of mi_make
    future_list = [dask_controller.client.submit(mi_make, i, X[:, i], scatter_y) for i in range(m1)]
    mi_list = [None] * len(future_list)
    for finished_future, future_return in distributed.as_completed(future_list, with_results=True):
        i, result_data = future_return
        mi_list[i] = result_data
        finished_future.cancel()

    # Clean up worker data by cancelling all the Futures
    dask_controller.client.cancel(scatter_y)

    # Convert the list of lists to an array
    mi = np.array(mi_list)
    assert (m1, m2) == mi.shape, "Array {sh} produced [({m1}, {m2}) expected]".format(sh=mi.shape, m1=m1, m2=m2)

    return mi
