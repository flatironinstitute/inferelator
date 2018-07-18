import numpy as np
import itertools
import math
import copy
import scipy.special

from inferelator_ng.utils import bool_to_index, nonzero_to_bool, make_array_2d
from inferelator_ng import utils


def bbsr(X, y, pp, weights, max_n):
    """

    :param X: np.ndarray [K x N]
        Predictor features
    :param y: np.ndarray [N,]
        Response variables
    :param pp: np.ndarray [K,]
        Predictors to model with
    :param weights: np.ndarray [K,]
        Weight matrix
    :param max_n: int
        Max number of predictors
    :return:
    """

    # Skip everything if there are no predictors in pp
    if pp.sum() == 0:
        return dict(pp=np.repeat(True, pp.shape[0]).tolist(), betas=0, betas_resc=0)

    # Subset data to desired predictors
    pp_idx = bool_to_index(pp)
    utils.Debug.vprint("Beginning regression with {pp_len} predictors".format(pp_len=len(pp_idx)), level=2)

    x = X[pp_idx, :].T
    gprior = weights[pp_idx].astype(np.dtype(float))

    # Make sure arrays are 2d
    make_array_2d(x)
    make_array_2d(y)
    make_array_2d(weights)

    # Reduce predictors

    pp[pp_idx] = (reduce_predictors(x, y, gprior, max_n))
    pp_idx = bool_to_index(pp)

    utils.Debug.vprint("Reduced to {pp_len} predictors".format(pp_len=len(pp_idx)), level=2)

    # Resubset with the newly reduced predictors
    x = X[pp_idx, :].T
    gprior = weights[pp_idx].astype(np.dtype(float))

    betas = best_subset_regression(x, y, gprior)
    utils.Debug.vprint("Calculated betas by BSR", level=2)

    betas_resc = predict_error_reduction(x, y, betas)
    utils.Debug.vprint("Calculated error reduction", level=2)

    return dict(pp=pp, betas=betas, betas_resc=betas_resc)


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
    combos = combo_index(k)  # [k x 2^k]
    bic_combos = calc_all_expected_BIC(x, y, gprior, combos)

    best_combo = combos[:, np.argmin(bic_combos)]
    best_betas = np.zeros(k, dtype=np.dtype(float))

    if best_combo.sum() > 0:
        idx = bool_to_index(best_combo)
        x = x[:, idx]
        beta_hat = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
        for i, j in enumerate(idx):
            best_betas[j] = beta_hat[i]

    return best_betas


def reduce_predictors(x, y, gprior, max_n):
    """
    Determine which predictors are the most valuable by calculating BICs for single and pairwise predictor models
    :param x: np.ndarray [n x k]
    :param y: np.ndarray [n x 1]
    :param gprior: [k x 1]
    :param max_n: int
    :return: np.ndarray [k,]
    """
    (_, k) = x.shape

    if k <= max_n:
        return np.ones(k, dtype=np.dtype(bool))
    else:
        # Get BIC for every combination of single or double predictors
        combos = np.hstack((np.diag(np.repeat(True, k)), select_index(k)))
        bic = calc_all_expected_BIC(x, y, gprior, combos)
        bic = np.multiply(combos.T, bic.reshape(-1, 1)).sum(axis=0)

        # Return a boolean index pointing to the lowest BIC predictors
        predictors = np.zeros(k, dtype=np.dtype(bool))
        predictors[np.argsort(bic)[0:max_n]] = True
        return predictors


def predict_error_reduction(x, y, betas):
    """

    :param x: np.ndarray [n x k]
    :param y: np.ndarray [n x 1]
    :param betas: list [k]
    :return:
    """
    (n, k) = x.shape
    pp_idx = bool_to_index(nonzero_to_bool(betas))

    ss_all = sigma_squared(x, y, betas)
    error_reduction = np.zeros(k, dtype=np.dtype(float))

    if len(pp_idx) == 1:
        error_reduction[pp_idx] = 1 - (ss_all / np.var(y, ddof=1))
        return error_reduction

    for pp_i in range(len(pp_idx)):
        leave_out = copy.copy(pp_idx)
        lost = leave_out.pop(pp_i)

        x_leaveout = x[:, leave_out]
        beta_hat = np.linalg.solve(np.dot(x_leaveout.T, x_leaveout), np.dot(x_leaveout.T, y))

        ss_leaveout = sigma_squared(x_leaveout, y, beta_hat)
        error_reduction[lost] = 1 - (ss_all / ss_leaveout)

    return error_reduction


def calc_all_expected_BIC(x, y, g, combinations):
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
    :return: np.ndarray [c,]
        Array of BICs corresponding to each combination
    """

    (n, k) = x.shape
    c = combinations.shape[1]

    # Sanity check the data
    assert n == y.shape[0]
    assert k == combinations.shape[0]

    # Precalculate xTx and xTy
    digamma_shape = scipy.special.digamma(n / 2)
    xtx = np.dot(x.T, x)  # [k x k]
    xty = np.dot(x.T, y)  # [k x 1]

    # Calculate the g-prior
    gprior = np.sqrt(1 / (g + 1)).reshape(-1, 1)
    gprior = np.multiply(gprior, gprior.T)

    bic = np.zeros(c, dtype=np.dtype(float))

    for i in range(c):
        # Check for a null model
        if combinations[:, i].sum() == 0:
            bic[i] = n * np.log(np.var(y, ddof=1))
            continue

        # Convert the boolean slice into an index
        p_c = np.where(combinations[:, i])[0]

        # Calculate the rate parameter from this specific combination of predictors
        rate = _calc_rate(x[:, p_c],
                          y,
                          xtx[:, p_c][p_c, :],
                          xty[p_c],
                          gprior[:, p_c][p_c, :])

        # Use the rate parameter to calculate the BIC for this combination of predictors
        if np.isinf(rate):
            bic[i] = rate
        else:
            bic[i] = n * (np.log(rate) - digamma_shape) + len(p_c) * np.log(n)

    return bic


def _calc_rate(x, y, xtx, xty, gprior):
    if np.linalg.matrix_rank(xtx, tol=1e-10) == xtx.shape[1]:
        # If xTx is full rank, regress xTx against xTy
        beta_hat = np.linalg.solve(xtx, xty)
        beta_flip = (0 - beta_hat.T)
        # Calculate the rate parameter for Zellner's g-prior
        rate = np.multiply(xtx, gprior)
        rate = np.dot(beta_flip, np.dot(rate, beta_flip.T))
        # Return the mean of the SSR and the rate parameter
        return (ssr(x, y, beta_hat) + rate) / 2
    else:
        # If xTx isn't full rank, return infinity
        return np.inf


def ssr(x, y, beta):
    """
    Sum of squared residuals (Y - XB)
    :param x: np.ndarray
    :param y: np.ndarray
    :param beta: np.ndarray
    :return: float
    """
    return np.square(np.subtract(y, np.dot(x, beta))).sum()


def sigma_squared(x, y, betas):
    return np.var(np.subtract(y, np.dot(x, betas).reshape(-1, 1)), ddof=1)


def combo_index(n):
    """
    Generate a boolean array that can mask to generate every possible combination of n objects
    :param n: int
        Number of objects
    :return: np.array
        [n x 2^n] array of bool
    """

    return np.array(map(list, itertools.product([False, True], repeat=n))).T


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

    combos = math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
    idx = np.array(map(list, itertools.combinations(range(n), r)))
    reindex = idx + np.array(range(idx.shape[0])).reshape(-1, 1) * n

    arr = np.full(combos * n, False, dtype=np.dtype(bool))
    arr[reindex.reshape(1, -1)] = True

    return arr.reshape((combos, n)).T
