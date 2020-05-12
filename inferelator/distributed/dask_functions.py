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


def amusr_regress_dask(X, Y, priors, prior_weight, n_tasks, genes, tfs, G, remove_autoregulation=True):
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
            x.append(x_df[k].get_gene_data(tf))  # list([N, K])
            y.append(y_data)
            tasks.append(k)  # [T,]

        prior = format_prior(prior, gene, tasks, prior_weight)
        return j, run_regression_EBIC(x, y, tf, tasks, gene, prior)

    def response_maker(y_df, i):
        y = []
        gene = genes[i]
        for k in range(n_tasks):
            if gene in y_df[k].gene_names:
                y.append((k, y_df[k].get_gene_data(gene, force_dense=True).reshape(-1, 1)))
        return y

    # Scatter common data to workers
    [scatter_x] = DaskController.client.scatter([X], broadcast=True, hash=False)
    [scatter_priors] = DaskController.client.scatter([priors], broadcast=True, hash=False)

    # Wait for scattering to finish before creating futures
    distributed.wait(scatter_x, timeout=DASK_SCATTER_TIMEOUT)
    distributed.wait(scatter_priors, timeout=DASK_SCATTER_TIMEOUT)

    future_list = [DaskController.client.submit(regression_maker, i, scatter_x, response_maker(Y, i), scatter_priors,
                                                tfs)
                   for i in range(G)]

    # Collect results as they finish instead of waiting for all workers to be done
    result_list = process_futures_into_list(future_list)

    DaskController.client.cancel(scatter_x)
    DaskController.client.cancel(scatter_priors)

    return result_list


def bbsr_regress_dask(X, Y, pp_mat, weights_mat, G, genes, nS):
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
        data = bayes_stats.bbsr(x, utils.scale_vector(y), pp[j, :].flatten(), weights[j, :].flatten(), nS)
        data['ind'] = j
        return j, data

    # Scatter common data to workers
    [scatter_x] = DaskController.client.scatter([X.values], broadcast=True, hash=False)
    [scatter_pp] = DaskController.client.scatter([pp_mat.values], broadcast=True, hash=False)
    [scatter_weights] = DaskController.client.scatter([weights_mat.values], broadcast=True, hash=False)

    # Wait for scattering to finish before creating futures
    distributed.wait(scatter_x, timeout=DASK_SCATTER_TIMEOUT)
    distributed.wait(scatter_pp, timeout=DASK_SCATTER_TIMEOUT)
    distributed.wait(scatter_weights, timeout=DASK_SCATTER_TIMEOUT)

    future_list = [DaskController.client.submit(regression_maker, i, scatter_x,
                                                Y.get_gene_data(i, force_dense=True).flatten(),
                                                scatter_pp, scatter_weights)
                   for i in range(G)]

    # Collect results as they finish instead of waiting for all workers to be done
    result_list = process_futures_into_list(future_list)

    DaskController.client.cancel(scatter_x)
    DaskController.client.cancel(scatter_pp)
    DaskController.client.cancel(scatter_weights)

    return result_list


def sklearn_regress_dask(X, Y, model, G, genes, min_coef):
    """
    Execute regression (SKLearn)

    :return: list
        Returns a list of regression results that the pileup_data can process
    """
    assert MPControl.is_dask()

    from inferelator.regression import sklearn_regression
    DaskController = MPControl.client

    def regression_maker(j, x, y):
        level = 0 if j % 100 == 0 else 2
        utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=genes[j], i=j, total=G), level=level)
        data = sklearn_regression.sklearn_gene(x, utils.scale_vector(y), copy.copy(model))
        data['ind'] = j
        return j, data

    # Scatter common data to workers
    [scatter_x] = DaskController.client.scatter([X.values], broadcast=True, hash=False)

    # Wait for scattering to finish before creating futures
    distributed.wait(scatter_x, timeout=DASK_SCATTER_TIMEOUT)

    future_list = [DaskController.client.submit(regression_maker, i, scatter_x,
                                                Y.get_gene_data(i, force_dense=True).flatten())
                   for i in range(G)]

    # Collect results as they finish instead of waiting for all workers to be done
    result_list = process_futures_into_list(future_list)

    DaskController.client.cancel(scatter_x)

    return result_list


def lasso_stars_regress_dask(X, Y, alphas, num_subsamples, random_seed, method, params, G, genes):
    """
    Execute regression (LASSO-StARS)

    :return: list
        Returns a list of regression results that the pileup_data can process
    """
    assert MPControl.is_dask()

    from inferelator.regression import stability_selection
    DaskController = MPControl.client

    def regression_maker(j, x, y):
        level = 0 if j % 100 == 0 else 2
        utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=genes[j], i=j, total=G), level=level)
        data = stability_selection.stars_model_select(x, utils.scale_vector(y), alphas, num_subsamples=num_subsamples,
                                                      method=method, random_seed=random_seed, **params)
        data['ind'] = j
        return j, data

    # Scatter common data to workers
    [scatter_x] = DaskController.client.scatter([X.values], broadcast=True, hash=False)

    # Wait for scattering to finish before creating futures
    distributed.wait(scatter_x, timeout=DASK_SCATTER_TIMEOUT)

    future_list = [DaskController.client.submit(regression_maker, i, scatter_x,
                                                Y.get_gene_data(i, force_dense=True).flatten())
                   for i in range(G)]

    # Collect results as they finish instead of waiting for all workers to be done
    result_list = process_futures_into_list(future_list)

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

    assert MPControl.is_dask()

    from inferelator.regression.mi import _calc_mi, _make_table, _make_discrete

    # Get a reference to the Dask controller
    DaskController = MPControl.client

    m1, m2 = X.shape[1], Y.shape[1]

    def mi_make(i, x, y):
        x = _make_discrete(x, bins)
        return i, [_calc_mi(_make_table(x, y[:, j], bins), logtype=logtype) for j in range(m2)]

    # Scatter Y to workers and keep track as Futures
    [scatter_y] = DaskController.client.scatter([Y], broadcast=True, hash=False)

    # Wait for scattering to finish before creating futures
    distributed.wait(scatter_y, timeout=DASK_SCATTER_TIMEOUT)

    # Build an asynchronous list of Futures for each calculation of mi_make
    future_list = [DaskController.client.submit(mi_make, i,
                                                X[:, i].A.flatten() if sps.isspmatrix(X) else X[:, i].flatten(),
                                                scatter_y)
                   for i in range(m1)]

    # Collect results as they finish instead of waiting for all workers to be done
    mi_list = process_futures_into_list(future_list)

    # Convert the list of lists to an array
    mi = np.array(mi_list)
    assert (m1, m2) == mi.shape, "Array {sh} produced [({m1}, {m2}) expected]".format(sh=mi.shape, m1=m1, m2=m2)

    DaskController.client.cancel(scatter_y)

    return mi


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

