import numpy as np
import pandas as pd

from scipy.special import comb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from inferelator import utils
from inferelator.utils import Validator as check

DEFAULT_prior_weight = 1.0
DEFAULT_Cs = np.logspace(np.log10(0.01), np.log10(10), 20)[::-1]

MAX_ITER = 1000
TOL = 1e-2
REL_TOL = None
MIN_WEIGHT_VAL = 0.1
MIN_RSS = 1e-10

OUT_DF_COLS = ['regulator', 'target', 'weights', 'resc_weights']


def run_regression_EBIC(
    X,
    Y,
    TFs,
    tasks,
    gene,
    prior,
    Cs=None,
    Ss=None,
    lambda_Bs=None,
    lambda_Ss=None,
    scale_data=False,
    return_lambdas=False,
    tol=TOL,
    rel_tol=REL_TOL,
    use_numba=False
):
    """
    Run multitask regression. Search the regularization coefficient space
    and select the model with the lowest eBIC.

    :param X: List of design matrixes for each task
    :type X: list(np.ndarray [N x K]) [t]
    :param Y: List consisting of response matrixes for each task
    :type Y: list(np.ndarray [N x 1]) [t]
    :param TFs: List of TF names for each task
    :type TFs: list(list(str) [K]) [t]
    :param tasks: List identifying each task
    :type tasks: list(int) [t]
    :param gene: The genes being modeled in each task
    :type gene: list(str) [t]
    :param prior:  The priors for this gene in a TF x Task array
    :type prior: np.ndarray [K x T]
    :param scale_data: Center and scale X and Y, defaults to False
    :type scale_data: bool
    :param return_lambdas: Return selected lambdaB and lambdaS,
        defaults to False
    :type return_lambdas: bool
    :param tol: Absolute convergence tolerance, defaults to 1e-2
    :type tol: float
    :param rel_tol: Relative convergence tolerance, defaults to None
    :type rel_tol: float, None
    :param use_numba: Use the numba-optimized S and B update functions
        for iterative search. Significant increase in speed, defaults to False
    :type use_numba: bool
    :return: dict
    """

    if use_numba:
        AMuSR_math.set_numba()

    assert len(X) == len(Y)
    assert len(X) == len(tasks)
    assert len(X) == len(gene)
    assert len(X) == len(TFs)

    assert prior.ndim == 2 if prior is not None else True
    assert prior.shape[1] == len(tasks) if prior is not None else True

    # The number of tasks
    n_tasks = len(X)

    # The number of predictors
    n_preds = X[0].shape[1]

    # A list of the number of samples for each task
    n_samples = [X[k].shape[0] for k in range(n_tasks)]

    ###### EBIC ######

    # Create empty block and sparse matrixes
    sparse_matrix = np.zeros((n_preds, n_tasks))
    block_matrix = np.zeros((n_preds, n_tasks))

    # Set the starting EBIC to infinity
    min_ebic = float('Inf')
    model_output = None

    # Calculate covariances
    X = scale_list_of_arrays(X) if scale_data else X
    Y = scale_list_of_arrays(Y) if scale_data else Y
    cov_C, cov_D = _covariance_by_task(X, Y)

    # Calculate lambda_B defaults if not provided
    if lambda_Bs is None:

        # Start with a default lambda B based on tasks, predictors, and samples
        lambda_B_param = np.sqrt((n_tasks * np.log(n_preds)) / np.mean(n_samples))

        # Modify by multiplying against the values in Cs
        lambda_Bs = lambda_B_param * np.array(DEFAULT_Cs if Cs is None else Cs)

    # Iterate through lambda_Bs
    for b in lambda_Bs:

        # Set scaling values if not provided
        Ss = np.linspace((1.0/n_tasks)+0.01, 0.99, 10)[::-1] if Ss is None else Ss

        # Iterate through lambda_Ss or calculate lambda Ss based on heuristic
        for s in lambda_Ss if lambda_Ss is not None else b * np.array(Ss):

            # Fit model
            combined_weights, sparse_matrix, block_matrix = amusr_fit(
                cov_C,
                cov_D,
                b,
                s,
                sparse_matrix,
                block_matrix,
                prior,
                tol=tol,
                rel_tol=rel_tol
            )

            # Score model
            ebic_score = ebic(X, Y, combined_weights, n_tasks, n_samples, n_preds)

            # Keep the model if it's the lowest scoring
            if ebic_score < min_ebic:
                min_ebic = ebic_score
                model_output = combined_weights
                opt_b, opt_s = b, s

    ###### RESCALE WEIGHTS ######
    output = {k: [] for k in tasks}

    if model_output is not None:
        for kx, k in enumerate(tasks):
            nonzero = model_output[:, kx] != 0
            if nonzero.sum() > 0:
                output[k].append(_final_weights(
                    X[kx][:, nonzero],
                    Y[kx],
                    np.asarray(TFs[kx])[nonzero],
                    gene[kx]
                ))

    output = {k: pd.concat(output[k], axis=0)
              for k in list(output.keys())
              if len(output[k]) > 0}

    return (output, opt_b, opt_s) if return_lambdas else output


def amusr_fit(
    cov_C,
    cov_D,
    lambda_B=0.,
    lambda_S=0.,
    sparse_matrix=None,
    block_matrix=None,
    prior=None,
    max_iter=MAX_ITER,
    tol=TOL,
    rel_tol=REL_TOL,
    rel_tol_min_iter=10,
    min_weight=MIN_WEIGHT_VAL
):
    """
    Fits regression model in which the weights matrix W (predictors x tasks)
    is decomposed in two components: B that captures block structure across tasks
    and S that allows for the differences.
    reference: Jalali et al., NIPS 2010. A Dirty Model for Multi-task Learning.
    :param cov_C: np.ndarray [T x K]
        Covariance of the predictors K to the response gene by task
    :param cov_D: np.ndarray [T x K x K]
        Covariance of the predictors K to K by task
    :param lambda_B: float
        Penalty coefficient for the block matrix
    :param lambda_S: float
        Penalty coefficient for the sparse matrix
    :param sparse_matrix: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are unique to each task
    :param block_matrix: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are shared between each task
    :param prior: np.ndarray [T x K]
        Matrix of known prior information
    :param max_iter: int
        Maximum number of model iterations if tol is not reached
    :param tol: float
        The tolerance for the stopping criteria (convergence)
    :param min_weight: float
        Regularize any weights below this threshold to 0
    :return combined_weights: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are the summation of both the task-specific
        model and the shared model
    :return sparse_matrix: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are unique to each task
    :return block_matrix: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are shared between each task
    """

    assert check.argument_type(lambda_B, (float, int, np.int64, np.float32))
    assert check.argument_type(lambda_S, (float, int, np.int64, np.float32))
    assert check.argument_type(max_iter, int)
    assert check.argument_type(tol, float)
    assert check.argument_type(min_weight, float)
    assert check.argument_numeric(rel_tol, allow_none=True)
    assert cov_C.shape[0] == cov_D.shape[0]
    assert cov_C.shape[1] == cov_D.shape[1]
    assert cov_D.shape[1] == cov_D.shape[2]
    assert prior.ndim == 2 if prior is not None else True

    n_tasks = cov_C.shape[0]
    n_features = cov_C.shape[1]

    assert check.argument_numeric(n_tasks, low=1)
    assert check.argument_numeric(n_features, low=1)

    # if S and B are provided -- warm starts -- will run faster
    if sparse_matrix is None or block_matrix is None:
        sparse_matrix = np.zeros((n_features, n_tasks))
        block_matrix = np.zeros((n_features, n_tasks))

    # If there is no prior for weights, create an array of 1s
    prior = np.ones((n_features, n_tasks)) if prior is None else prior

    # Initialize weights
    combined_weights = sparse_matrix + block_matrix


    iter_tols = np.zeros(max_iter)
    for i in range(max_iter):

        # Keep a reference to the old combined_weights
        _combined_weights_old = combined_weights

        # Update sparse and block-sparse coefficients
        sparse_matrix = AMuSR_math.updateS(
            cov_C,
            cov_D,
            block_matrix,
            sparse_matrix,
            lambda_S,
            prior,
            n_tasks,
            n_features
        )

        block_matrix = AMuSR_math.updateB(
            cov_C,
            cov_D,
            block_matrix,
            sparse_matrix,
            lambda_B,
            prior,
            n_tasks,
            n_features
        )

        # Weights matrix (W) is the sum of a sparse (S) and a block-sparse (B) matrix
        combined_weights = sparse_matrix + block_matrix

        # If convergence tolerance reached, break loop and move on
        iter_tols[i] = np.max(np.abs(combined_weights - _combined_weights_old))

        if iter_tols[i] < tol:
            break

        # If the maximum over the last few iterations is less than the relative tolerance, break loop and move on
        if rel_tol is not None and (i > rel_tol_min_iter):
            lb_start, lb_stop = i - rel_tol_min_iter, i
            iter_rel_max = iter_tols[lb_start: lb_stop] - iter_tols[lb_start - 1: lb_stop - 1]
            if np.max(iter_rel_max) < rel_tol:
                break

    # Set small values of W to zero
    # Since we don't run the algorithm until update equals zero
    combined_weights[np.abs(combined_weights) < min_weight] = 0

    return combined_weights, sparse_matrix, block_matrix

def _covariance_by_task(X, Y):
    """
    Returns C and D, containing terms for covariance update for OLS fit
    C: transpose(X_j)*Y for each feature j
    D: transpose(X_j)*X_l for each feature j for each feature l
    Reference: Friedman, Hastie, Tibshirani, 2010 in Journal of Statistical Software
    Regularization Paths for Generalized Linear Models via Coordinate Descent
    :param X: list(np.ndarray [N x K]) [T]
        List of design values for each task. Must be aligned on the feature (K) axis.
    :param Y: list(np.ndarray [N x 1]) [T]
        List of response values for each task
    :return cov_C, cov_D: np.ndarray [T x K], np.ndarray [T x K x K]
        Covariance of the predictors K to the response gene by task
        Covariance of the predictors K to K by task
    """

    assert check.argument_type(X, list)
    assert check.argument_type(Y, list)
    assert len(X) == len(Y)
    assert max([xk.shape[1] for xk in X]) == min([xk.shape[1] for xk in X])

    # Calculate dimensionality for returned arrays
    n_tasks = len(X)
    n_features = max([xk.shape[1] for xk in X])

    # Build empty arrays
    cov_C = np.zeros((n_tasks, n_features))
    cov_D = np.zeros((n_tasks, n_features, n_features))

    # Populate arrays
    for task_id in range(n_tasks):
        cov_C[task_id] = np.dot(Y[task_id].transpose(), X[task_id])  # yTx
        cov_D[task_id] = np.dot(X[task_id].transpose(), X[task_id])  # xTx

    return cov_C, cov_D


def sum_squared_errors(X, Y, W, k):
    '''
    Get RSS for a particular task 'k'
    '''
    return(np.sum((Y[k].T-np.dot(X[k], W[:,k]))**2))


def ebic(X, Y, model_weights, n_tasks, n_samples, n_preds, gamma=1, min_rss=MIN_RSS):
    """
    Calculate EBIC for each task, and take the mean
    Extended Bayesian information criteria for model selection with large model spaces
    https://doi.org/10.1093/biomet/asn034
    :param X: list([N x K]) [T]
        List consisting of design matrixes for each task
    :param Y: list([N x 1]) [T]
        List consisting of response matrixes for each task
    :param model_weights: np.ndarray [K x T]
        Fit model coefficients for each task
    :param n_tasks: int
        Number of tasks T
    :param n_samples: list(int) [T]
        Number of samples for each task
    :param n_preds: int
        Number of predictors
    :param gamma: float
        Gamma parameter for extended BIC
    :param min_rss: float
        Floor value for RSS to prevent log(0)
    :return: float
        Mean ebic for all tasks
    """

    assert check.argument_type(X, list)
    assert check.argument_type(Y, list)
    assert check.argument_type(model_weights, np.ndarray)
    assert check.argument_type(n_tasks, int)
    assert check.argument_type(n_samples, list)
    assert check.argument_type(n_preds, int)
    assert check.argument_numeric(n_tasks, low=1)
    assert check.argument_numeric(n_preds, low=1)
    assert check.argument_type(gamma, (float, int))

    EBIC = []

    for task_id in range(n_tasks):
        # Get the number of samples for this task
        task_samples = n_samples[task_id]
        task_model = model_weights[:, task_id]

        # Find the number of non-zero predictors
        nonzero_pred = (task_model != 0).sum()

        # Calculate RSS for the task model
        rss = sum_squared_errors(X, Y, model_weights, task_id)

        # Calculate bayes information criterion using likelihood = RSS / n
        # Calculate the first component of BIC with a non-zero floor for RSS (so BIC is always finite)
        bic = task_samples * np.log(rss / task_samples) if rss > 0 else task_samples * np.log(min_rss / task_samples)

        # Calculate the second component of BIC
        bic_penalty = nonzero_pred * np.log(task_samples)

        # Calculate the extended component of eBIC
        # 2 * gamma * ln(number of non-zero predictor combinations of all predictors)
        bic_extension = 2 * gamma * np.log(comb(n_preds, nonzero_pred))

        # Combine all the components and put them in a list
        EBIC.append(bic + bic_penalty + bic_extension)

    return np.mean(EBIC)

def _final_weights(X, y, TFs, gene):
    """
    returns reduction on variance explained for each predictor
    (model without each predictor compared to full model)
    see: Greenfield et al., 2013. Robust data-driven incorporation of prior
    knowledge into the inference of dynamic regulatory networks.
    :param X: np.ndarray [N x k]
        A design matrix with N samples and k non-zero predictors
    :param y: np.ndarray [N x 1]
        A response matrix with N samples of a specific gene expression
    :param TFs: list() or np.ndarray or pd.Series
        A list of non-zero TFs (k) included in the model
    :param gene: str
        The gene modeled
    :return out_weights: pd.DataFrame [k x 4]
        An edge table (regulator -> target) with the model coefficient and the variance explained by that predictor for
        each non-zero predictor
    """

    assert check.argument_type(X, np.ndarray)
    assert check.argument_type(y, np.ndarray)
    assert check.argument_type(TFs, (list, np.ndarray, pd.Series))

    n_preds = len(TFs)

    # Linear fit using sklearn
    ols = LinearRegression().fit(X, y)

    # save weights and initialize rescaled weights vector
    weights = ols.coef_[0]
    resc_weights = np.zeros(n_preds)

    # variance of residuals (full model)
    var_full = np.var((y - ols.predict(X)) ** 2)

    # when there is only one predictor
    if n_preds == 1:
        resc_weights[0] = 1 - (var_full / np.var(y))
    # remove each at a time and calculate variance explained
    else:
        for j in range(len(TFs)):
            X_noj = X[:, np.setdiff1d(range(n_preds), j)]
            ols = LinearRegression().fit(X_noj, y)
            var_noj = np.var((y - ols.predict(X_noj)) ** 2)
            resc_weights[j] = 1 - (var_full / var_noj)

    # Format output into an edge table
    out_weights = pd.DataFrame([TFs, [gene] * len(TFs), weights, resc_weights]).transpose()
    out_weights.columns = OUT_DF_COLS

    return out_weights


def scale_list_of_arrays(X):
    """
    Scale a list of arrays so that each has mean 0 and unit variance

    :param X: list(np.ndarray) [T]
    :return X: list(np.ndarray) [T]
    """

    assert check.argument_type(X, list)

    return [StandardScaler().fit_transform(xk.astype(float)) for xk in X]


class AMuSR_math:

    _numba = False

    @staticmethod
    def updateS(C, D, B, S, lamS, prior, n_tasks, n_features):
        """
        returns updated coefficients for S (predictors x tasks)
        lasso regularized -- using cyclical coordinate descent and
        soft-thresholding
        """
        # update each task independently (shared penalty only)
        for k in range(n_tasks):

            c = C[k]
            d = D[k]

            b = B[:, k]
            s = S[:, k]
            p = prior[:, k] * lamS

            # cycle through predictors

            for j in range(n_features):
                # set sparse coefficient for predictor j to zero
                s[j] = 0.

                # calculate next coefficient based on fit only
                if d[j,j] == 0:
                    alpha = 0.
                else:
                    alpha = (c[j]- np.sum((b + s) * d[j])) / d[j,j]

                # lasso regularization
                if abs(alpha) <= p[j]:
                    s[j] = 0.
                else:
                    s[j] = alpha - (np.sign(alpha) * p[j])

            # update current task
            S[:, k] = s

        return S

    @staticmethod
    def updateB(C, D, B, S, lamB, prior, n_tasks, n_features):
        """
        returns updated coefficients for B (predictors x tasks)
        block regularized (l_1/l_inf) -- using cyclical coordinate descent and
        soft-thresholding on the l_1 norm across tasks
        reference: Liu et al, ICML 2009. Blockwise coordinate descent procedures
        for the multi-task lasso, with applications to neural semantic basis discovery.
        """

        # cycles through predictors
        for j in range(n_features):

            # initialize next coefficients
            alphas = np.zeros(n_tasks)

            # update tasks for each predictor together
            d = D[:, :, j]

            for k in range(n_tasks):

                d_kjj = d[k, j]

                if d_kjj == 0:

                    alphas[k] = 0

                else:

                    # get previous block-sparse
                    # copies because B is C-ordered
                    b = B[:, k]

                    # set block-sparse coefficient for feature j to zero
                    b[j] = 0.

                    # calculate next coefficient based on fit only
                    alphas[k] = (C[k, j] - np.sum((b + S[:, k]) * d[k, :])) / d_kjj


            # set all tasks to zero if l1-norm less than lamB
            if np.linalg.norm(alphas, 1) <= lamB:
                B[j, :] = np.zeros(n_tasks)

            # regularized update for predictors with larger l1-norm
            else:
                # find number of coefficients that would make l1-norm greater than penalty
                indices = np.abs(alphas).argsort()[::-1]
                sorted_alphas = alphas[indices]

                m_star = np.argmax((np.abs(sorted_alphas).cumsum() - lamB) / (np.arange(n_tasks) + 1))

                # initialize new weights
                new_weights = np.zeros(n_tasks)

                # keep small coefficients and regularize large ones (in above group)
                for k in range(n_tasks):

                    idx = indices[k]

                    if k > m_star:
                        new_weights[idx] = sorted_alphas[k]

                    else:
                        sign = np.sign(sorted_alphas[k])
                        update_term = np.sum(np.abs(sorted_alphas)[:m_star + 1]) - lamB
                        new_weights[idx] = (sign/(m_star + 1)) * update_term

                # update current predictor
                B[j, :] = new_weights

        return B

    @classmethod
    def set_numba(cls):

        # If this has already been called, skip
        if cls._numba:
            return

        else:

            # If we can't import numba, skip (and set a flag so we don't try again)
            try:
                import numba

            except ImportError:
                utils.Debug.vprint("Unable to import numba; using python-native functions instead", level=0)
                cls._numba = True
                return

            utils.Debug.vprint("Using numba functions for AMuSR regression", level=0)

            # Replace the existing functions with JIT functions
            cls.updateB = numba.jit(cls.updateB, nopython=True)
            cls.updateS = numba.jit(cls.updateS, nopython=True)
            cls._numba = True
