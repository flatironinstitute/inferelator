from __future__ import division

import numpy as np
import itertools
import math
import scipy.special

from inferelator import utils
from inferelator.regression import base_regression


def bbsr(X, y, pp, weights, max_k):
    """
    Run BBSR to regress a response variable y in n conditions against predictors X in n conditions. Use the prior
    predictors matrix to filter the number of predictors from something massive to max_k.
    :param X: np.ndarray [K x N]
        Predictor features
    :param y: np.ndarray [N,]
        Response variables
    :param pp: np.ndarray [K,]
        Predictors to model with
    :param weights: np.ndarray [K,]
        Weight matrix
    :param max_k: int
        Max number of predictors
    :return: dict
        pp: Boolean array indicating which predictors are included in the model                 [K,]
        betas: Float array indicating the beta for each predictor included in the model         [K,]
        betas_resc: Float array indicating how much each predictor is contributing to the model [K,]
    """

    # Skip everything if there are no predictors in pp
    if pp.sum() == 0:
        return dict(pp=np.repeat(True, pp.shape[0]).tolist(),
                    betas=np.zeros(pp.shape[0]),
                    betas_resc=np.zeros(pp.shape[0]))

    # Subset data to desired predictors
    pp_idx = base_regression.bool_to_index(pp)
    utils.Debug.vprint("Beginning regression with {pp_len} predictors".format(pp_len=len(pp_idx)), level=2)

    x = X[pp_idx, :].T
    gprior = weights[pp_idx].astype(np.dtype(float))

    # Make sure arrays are 2d
    utils.make_array_2d(x)
    utils.make_array_2d(y)
    utils.make_array_2d(gprior)

    # Reduce predictors to max_k
    pp[pp_idx] = reduce_predictors(x, y, gprior, max_k)
    pp_idx = base_regression.bool_to_index(pp)

    utils.Debug.vprint("Reduced to {pp_len} predictors".format(pp_len=len(pp_idx)), level=2)

    # Resubset with the newly reduced predictors
    x = X[pp_idx, :].T
    gprior = weights[pp_idx].astype(np.dtype(float))
    utils.make_array_2d(gprior)

    betas = best_subset_regression(x, y, gprior)
    betas_resc = base_regression.predict_error_reduction(x, y, betas)

    return dict(pp=pp,
                betas=betas,
                betas_resc=betas_resc)


def best_subset_regression(x, y, gprior):
    """

    :param x: np.ndarray
        Independent (predictor) variables [n x k]
    :param y: np.ndarray
        Dependent (response) variable [n x 1]
    :param gprior: np.ndarray
        Weighted priors [k x 1]
    :return:
    """
    (n, k) = x.shape
    combos = combo_index(k)

    bic_combos = calc_all_expected_BIC(x, y, gprior, combos, check_rank=False)

    best_betas = np.zeros(k, dtype=np.dtype(float))
    try:
        best_combo = combos[:, _best_combo_idx(x, bic_combos, combos)]
    except np.linalg.LinAlgError:
        return best_betas

    if best_combo.sum() > 0:
        best_betas = base_regression.recalculate_betas_from_selected(x, y, best_combo)

    return best_betas


def reduce_predictors(x, y, gprior, max_k):
    """
    Determine which predictors are the most valuable by calculating BICs for single and pairwise predictor models
    :param x: np.ndarray [n x k]
    :param y: np.ndarray [n x 1]
    :param gprior: [k x 1]
    :param max_k: int
    :return: np.ndarray [k,]
    """
    (_, k) = x.shape

    if k <= max_k:
        return np.ones(k, dtype=np.dtype(bool))
    else:
        # Get BIC for every combination of single or double predictors
        combos = np.hstack((np.diag(np.repeat(True, k)), select_index(k)))
        bic = calc_all_expected_BIC(x, y, gprior, combos)

        reset = np.seterr(divide='ignore', invalid='ignore')
        bic = np.multiply(combos.T, bic.reshape(-1, 1)).sum(axis=0)

        # Return a boolean index pointing to the lowest BIC predictors
        predictors = np.zeros(k, dtype=np.dtype(bool))
        predictors[np.argsort(bic)[0:max_k]] = True

        np.seterr(**reset)
        return predictors


def calc_all_expected_BIC(x, y, g, combinations, check_rank=True):
    """
    Calculate BICs for every combination of predictors given in combinations
    :param x: np.ndarray [n x k]
        Array of predictor data
    :param y: np.ndarray [n x 1]
        Array of response data
    :param g: np.ndarray [k x 1]
        Weights for predictors
    :param combinations: np.ndarray [k x c]
        Combinations of predictors to try; each combination should be booleans with a length corresponding to the
        number of predictors
    :param check_rank: bool
        Explicitly check to see that xTx is nonsingular for every combination. If false, will only catch singular xTx
        that causes np.linalg.solve to throw an exception
    :return: np.ndarray [c,]
        Array of BICs corresponding to each combination
    """
    (n, k) = x.shape
    c = combinations.shape[1]

    # Sanity check the data
    assert n == y.shape[0]
    assert k == combinations.shape[0]

    # Precalculate xTx and xTy
    digamma_shape = scipy.special.digamma(n / 2.0)
    xtx = np.dot(x.T, x)  # [k x k]
    xty = np.dot(x.T, y)  # [k x 1]

    # Calculate the g-prior
    gprior = np.repeat(np.sqrt(1.0 / (g + 1.0)), k, axis=1)
    gprior = np.multiply(gprior, gprior.T)
    bic = np.zeros(c, dtype=np.dtype(float))

    for i in range(c):
        # Convert the boolean slice into an index
        c_idx = base_regression.bool_to_index(combinations[:, i])
        k_included = len(c_idx)

        # Check for a null model
        if k_included == 0:
            bic[i] = n * np.log(np.var(y, ddof=1))
            continue

        # Calculate the rate parameter from this specific combination of predictors
        try:
            rate = _calc_rate(x[:, c_idx],
                              y,
                              xtx[:, c_idx][c_idx, :],
                              xty[c_idx],
                              gprior[:, c_idx][c_idx, :],
                              check_rank=check_rank)
            if np.isfinite(rate) and rate > 0:
                bic[i] = n * (np.log(rate) - digamma_shape) + k_included * np.log(n)
            else:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            bic[i] = np.inf
    return bic


def _calc_rate(x, y, xtx, xty, gprior, check_rank=True):
    # Check to see if xTx is nonsingular (if necessary)
    if check_rank and not _matrix_full_rank(xtx):
        raise np.linalg.LinAlgError

    # Regress xTx against xTy and calculate the SSR
    beta_hat = np.linalg.solve(xtx, xty)
    ssr_beta_hat = ssr(x, y, beta_hat)

    # Calculate the rate parameter for Zellner's g-prior
    beta_flip = (0 - beta_hat.T)
    rate = xtx * gprior
    rate = np.dot(rate, beta_flip.T)
    rate = np.dot(beta_flip, rate)

    # Return the mean of the SSR and the rate parameter
    return (ssr_beta_hat + rate) / 2


def _best_combo_idx(x, bic, combo):
    """
    Find the lowest BIC combination that comes from a nonsingular xTx
    :param x: [n x k]
    :param bic: [c,] array of floats
    :param combo: [k x c]
    :return:
    """

    for i in range(combo.shape[1]):
        bic_idx = np.argmin(bic)  # In case of a tie, np.argmin returns the leftmost index
        c = combo[:, bic_idx]

        if c.sum() == 0:
            return bic_idx

        x_slice = x[:, c]
        if _matrix_full_rank(np.dot(x_slice.T, x_slice)):
            return bic_idx
        else:
            bic[bic_idx] = np.inf

    raise np.linalg.LinAlgError


def _matrix_full_rank(mat, tol=1e-10):
    return np.linalg.matrix_rank(mat, tol=tol) == mat.shape[1]


def ssr(x, y, beta):
    """
    Sum of squared residuals (Y - XB)
    :param x: np.ndarray [N x M]
    :param y: np.ndarray [N x P]
    :param beta: np.ndarray [M x P]
    :return: float
    """
    assert x.shape[1] == beta.shape[0]

    resid = y - np.dot(x, beta)
    return (resid * resid).sum()


def combo_index(n):
    """
    Generate a boolean array that can mask to generate every possible combination of n objects
    :param n: int
        Number of objects
    :return: np.array
        [n x 2^n] array of bool
    """
    assert n >= 0

    return np.array(list(itertools.product([False, True], repeat=n))).T


def select_index(n, r=2):
    """
    Generate a boolean array that can mask to generate every selection of r objects from a total pool of n objects
    :param n: int
        Number of objects
    :param r: int
        Number of objects to select
    :return: np.arrap
        [n x n!/(r!(n-r)!)] array of bool
    """
    assert n >= 0

    combos = int(math.factorial(n) / (math.factorial(r) * math.factorial(n - r)))
    idx = np.array(list(itertools.combinations(range(n), r)))
    idx = idx + np.array(range(idx.shape[0])).reshape(-1, 1) * n

    arr = np.full(combos * n, False, dtype=np.dtype(bool))
    arr[idx.reshape(1, -1)] = True

    return arr.reshape((combos, n)).T
