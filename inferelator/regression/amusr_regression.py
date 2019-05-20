import numpy as np
import pandas as pd
import itertools

from scipy.special import comb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from inferelator.distributed.inferelator_mp import MPControl
from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.regression import base_regression

# Shadow built-in zip with itertools.izip if this is python2 (This puts out a memory dumpster fire)
try:
    from itertools import izip as zip
except ImportError:
    pass

MAX_ITER = 1000
TOL = 1e-2
MIN_WEIGHT_VAL = 0.1
MIN_RSS = 1e-10


class AMuSR_regression(base_regression.BaseRegression):
    X = None  # list(pd.DataFrame [N, K])
    Y = None  # list(pd.DataFrame [N, G])
    priors = None  # list(pd.DataFrame [G, K]) OR pd.DataFrame [G, K]
    tfs = None  # pd.Index OR list
    genes = None  # pd.Index OR list

    K = None  # int
    G = None  # int
    n_tasks = None  # int
    prior_weight = 1.0  # float
    remove_autoregulation = True  # bool

    def __init__(self, X, Y, tfs=None, genes=None, priors=None, prior_weight=1, remove_autoregulation=True):
        """
        Set up a regression object for multitask regression

        :param X: list(pd.DataFrame [N, K])
        :param Y: list(pd.DataFrame [N, G])
        :param priors: pd.DataFrame [G, K]
        :param prior_weight: float
        :param remove_autoregulation: bool
        """

        # Set the data into the regression object
        self.X = X
        self.Y = Y
        self.n_tasks = len(X)

        # Set the priors and weight into the regression object
        self.priors = priors
        self.prior_weight = float(prior_weight)

        # Construct a list of TFs & genes if they are not passed in
        if tfs is None or genes is None:
            tfs, genes = [], []
            for design, response in zip(X, Y):
                tfs.append(design.columns)
                genes.append(response.columns)
            self.tfs = filter_genes_on_tasks(tfs, "intersection")
            self.genes = filter_genes_on_tasks(genes, "intersection")
        else:
            self.tfs, self.genes = tfs, genes

        # Set the regulators and targets into the regression object
        self.K, self.G = len(tfs), len(genes)
        self.remove_autoregulation = remove_autoregulation

    def regress(self):
        """
        Execute multitask (AMUSR)

        :return: list
            Returns a list of regression results that the amusr_regression pileup_data can process
        """

        if MPControl.is_dask():
            from inferelator.distributed.dask_functions import amusr_regress_dask
            return amusr_regress_dask(self.X, self.Y, self.priors, self.prior_weight, self.n_tasks, self.genes,
                                      self.tfs, self.G, remove_autoregulation=self.remove_autoregulation)

        def regression_maker(j):
            level = 0 if j % 100 == 0 else 2
            utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=self.genes[j], i=j, total=self.G),
                                 level=level)

            gene = self.genes[j]
            x, y, tasks = [], [], []

            if self.remove_autoregulation:
                tfs = [t for t in self.tfs if t != gene]
            else:
                tfs = self.tfs

            for k in range(self.n_tasks):
                if gene in self.Y[k]:
                    x.append(self.X[k].loc[:, tfs].values)  # list([N, K])
                    y.append(self.Y[k].loc[:, gene].values.reshape(-1, 1))  # list([N, 1])
                    tasks.append(k)  # [T,]

            prior = _format_prior(self.priors, gene, tasks, self.prior_weight)
            return run_regression_EBIC(x, y, tfs, tasks, gene, prior)

        return MPControl.map(regression_maker, range(self.G))

    def pileup_data(self, run_data):

        weights = []
        rescaled_weights = []

        for k in range(self.n_tasks):
            results_k = []
            for res in run_data:
                try:
                    results_k.append(res[k])
                except KeyError:
                    pass

            results_k = pd.concat(results_k)
            weights_k = _format_weights(results_k, 'weights', self.genes, self.tfs)
            rescaled_weights_k = _format_weights(results_k, 'resc_weights', self.genes, self.tfs)
            rescaled_weights_k[rescaled_weights_k < 0.] = 0

            weights.append(weights_k)
            rescaled_weights.append(rescaled_weights_k)

        return weights, rescaled_weights


def run_regression_EBIC(X, Y, TFs, tasks, gene, prior):
    """
    Run multitask regression

    :param X: list(np.ndarray [N x K]) [T]
        List consisting of design matrixes for each task
    :param Y: list(np.ndarray [N x 1]) [T]
        List consisting of response matrixes for each task
    :param TFs: list [K]
        List of TF names for each feature
    :param tasks: list(int) [T]
        List identifying each task
    :param gene: str
        The gene being modeled
    :param prior: np.ndarray [K x T]
        The priors for this gene in a TF x Task array
    :return output: dict
        A dict, keyed by task, containing a dataframe with model coefficients in edge (regulator -> target) format
    """

    assert check.argument_type(X, list)
    assert check.argument_type(Y, list)
    assert check.argument_type(TFs, list)
    assert check.argument_type(tasks, list)
    assert len(X) == len(Y)
    assert len(X) == len(tasks)
    assert max([xk.shape[1] for xk in X]) == min([xk.shape[1] for xk in X])
    assert min([xk.shape[0] for xk in X]) > 0
    assert all([xk.shape[0] == yk.shape[0] for xk, yk in zip(X, Y)])

    n_tasks = len(X)  # The number of tasks
    n_preds = X[0].shape[1]  # The number of predictors
    n_samples = [xk.shape[0] for xk in X]  # A list of the number of samples for each task

    ###### EBIC ######

    # Set a grid search space for the lambda parameters
    Cs = np.logspace(np.log10(0.01), np.log10(10), 20)[::-1]
    Ss = np.linspace((1.0 / n_tasks) + 0.01, 0.99, 10)[::-1]  # in paper I used 0.51 as minimum for all networks

    # Start with a default lambda B based on tasks, predictors, and samples
    lambda_B_param = np.sqrt((n_tasks * np.log(n_preds)) / np.mean(n_samples))

    # Center and scale the data
    X = scale_list_of_arrays(X)
    Y = scale_list_of_arrays(Y)

    # Calculate covariances
    cov_C, cov_D = _covariance_by_task(X, Y)

    # Create empty block and sparse matrixes
    sparse_matrix = np.zeros((n_preds, n_tasks))
    block_matrix = np.zeros((n_preds, n_tasks))

    # Set the starting EBIC to infinity
    min_ebic = float('Inf')
    model_output = None

    # Search lambda space for the model with the lowest EBIC
    for c, s in itertools.product(Cs, Ss):
        lambda_B_tmp = c * lambda_B_param
        lambda_S_tmp = s * lambda_B_tmp
        combined_weights, sparse_matrix, block_matrix = amusr_fit(X, Y, lambda_B_tmp, lambda_S_tmp, cov_C, cov_D,
                                                                  sparse_matrix, block_matrix, prior)
        ebic_score = ebic(X, Y, combined_weights, n_tasks, n_samples, n_preds)
        if ebic_score < min_ebic:
            model_output = combined_weights

    # Rescale weights for output.
    # This will be processed later by AMuSR_regression.pileup_data()
    output = {}

    if model_output is not None:
        for kx, k in enumerate(tasks):
            nonzero = model_output[:, kx] != 0
            if nonzero.sum() > 0:
                included_tfs = np.asarray(TFs)[nonzero]
                output[k] = _final_weights(X[kx][:, nonzero], Y[kx], included_tfs, gene)

    return output


def amusr_fit(X, Y, lambda_B=0., lambda_S=0., cov_C=None, cov_D=None, sparse_matrix=None, block_matrix=None, prior=None,
              max_iter=MAX_ITER, tol=TOL, min_weight=MIN_WEIGHT_VAL):
    """
    Fits regression model in which the weights matrix W (predictors x tasks)
    is decomposed in two components: B that captures block structure across tasks
    and S that allows for the differences.
    reference: Jalali et al., NIPS 2010. A Dirty Model for Multi-task Learning.

    :param X: list(np.ndarray [N x K]) [T]
        List of design values for each task. Must be aligned on the feature (K) axis.
    :param Y: list(np.ndarray [N x 1]) [T]
        List of response values for each task
    :param lambda_B: float
        Penalty coefficient for the block matrix
    :param lambda_S: float
        Penalty coefficient for the sparse matrix
    :param cov_C: np.ndarray [T x K]
        Covariance of the predictors K to the response gene by task
    :param cov_D: np.ndarray [T x K x K]
        Covariance of the predictors K to K by task
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
    :return combined_weights, sparse_matrix, block_matrix: np.ndarray [K x T], np.ndarray [K x T], np.ndarray [K x T]
    """

    assert check.argument_type(X, list)
    assert check.argument_type(Y, list)
    assert check.argument_type(lambda_B, (float, int))
    assert check.argument_type(lambda_S, (float, int))
    assert check.argument_type(max_iter, int)
    assert check.argument_type(tol, float)
    assert check.argument_type(min_weight, float)
    assert len(X) == len(Y)
    assert max([xk.shape[1] for xk in X]) == min([xk.shape[1] for xk in X])

    n_tasks = len(X)
    n_features = max([xk.shape[1] for xk in X])

    # calculate covariance update terms if not provided
    if cov_C is None or cov_D is None:
        cov_C, cov_D = _covariance_by_task(X, Y)

    assert cov_C.shape[0] == cov_D.shape[0]
    assert cov_C.shape[1] == cov_D.shape[1]

    # if S and B are provided -- warm starts -- will run faster
    if sparse_matrix is None or block_matrix is None:
        sparse_matrix = np.zeros((n_features, n_tasks))
        block_matrix = np.zeros((n_features, n_tasks))

    # If there is no prior for weights, create an array of 1s
    prior = np.ones((n_features, n_tasks)) if prior is None else prior

    # Initialize weights
    combined_weights = sparse_matrix + block_matrix
    for _ in range(max_iter):

        # Save old weights (to check convergence)
        old_weights = np.copy(combined_weights)

        # Update sparse and block coefficients
        sparse_matrix = _update_sparse(cov_C, cov_D, block_matrix, sparse_matrix, lambda_S, prior)
        block_matrix = _update_block(cov_C, cov_D, block_matrix, sparse_matrix, lambda_S)

        # Combine sparse matrix and block matrix to a unified weight matrix
        combined_weights = sparse_matrix + block_matrix

        # If convergence tolerance reached, break loop and move on
        if np.max(np.abs(combined_weights - old_weights)) < tol:
            break

    # Weights matrix (W) is the sum of a sparse (S) and a block-sparse (B) matrix
    combined_weights = sparse_matrix + block_matrix

    # Set small values of W to zero
    # Since we don't run the algorithm until update equals zero
    combined_weights[np.abs(combined_weights) < min_weight] = 0

    return combined_weights, sparse_matrix, block_matrix


def scale_list_of_arrays(X):
    """
    Scale a list of data frames so that each has mean 0 and unit variance
    :param X: list(np.ndarray) [T]
    :return X: list(np.ndarray) [T]
    """

    assert check.argument_type(X, list)

    return [StandardScaler().fit_transform(xk) for xk in X]


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


def _update_sparse(cov_C, cov_D, block_matrix, sparse_matrix, lambda_S, prior):
    """
    returns updated coefficients for S (predictors x tasks)
    lasso regularized -- using cyclical coordinate descent and
    soft-thresholding
    :param cov_C: np.ndarray [T x K]
        Covariance of the predictors K to the response gene by task
    :param cov_D: np.ndarray [T x K x K]
        Covariance of the predictors K to K by task
    :param sparse_matrix: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are unique to each task
    :param block_matrix: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are shared between each task
    :param lambda_S: float
        Penalty coefficient for the sparse matrix
    :param prior: np.ndarray [T x K]
        Matrix of known prior information
    :return:
    """

    assert cov_C.shape[0] == cov_D.shape[0]
    assert cov_C.shape[1] == cov_D.shape[1]
    assert block_matrix.shape == sparse_matrix.shape
    assert check.argument_type(lambda_S, (float, int))

    n_features = cov_C.shape[1]
    n_tasks = cov_C.shape[0]

    # Update each task independently (shared penalty only)
    for task_id in range(n_tasks):
        # Task covariance update terms
        task_c = cov_C[task_id]
        task_d = cov_D[task_id]

        # Previous task block-sparse and sparse coefficients
        task_b = block_matrix[:, task_id]
        task_s = sparse_matrix[:, task_id]
        task_prior = prior[:, task_id]

        # Cycle through predictors
        for j in range(n_features):
            # Set sparse coefficient for predictor j to zero
            sparse_tmp = np.copy(task_s)
            sparse_tmp[j] = 0.

            # Calculate next coefficient based on fit only
            alpha = (task_c[j] - np.sum((task_b + sparse_tmp) * task_d[j])) / task_d[j, j] if task_d[j, j] != 0 else 0.

            # Lasso regularization
            if abs(alpha) > task_prior[j]:
                task_s[j] = alpha - (np.sign(alpha) * task_prior[j] * lambda_S)
            else:
                task_s[j] = 0

        # update current task
        sparse_matrix[:, task_id] = task_s

    return sparse_matrix


def _update_block(cov_C, cov_D, block_matrix, sparse_matrix, lambda_B):
    """
    returns updated coefficients for B (predictors x tasks)
    block regularized (l_1/l_inf) -- using cyclical coordinate descent and
    soft-thresholding on the l_1 norm across tasks
    reference: Liu et al, ICML 2009. Blockwise coordinate descent procedures
    for the multi-task lasso, with applications to neural semantic basis discovery.
    :param cov_C: np.ndarray [T x K]
        Covariance of the predictors K to the response gene by task
    :param cov_D: np.ndarray [T x K x K]
        Covariance of the predictors K to K by task
    :param sparse_matrix: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are unique to each task
    :param block_matrix: np.ndarray [K x T]
        Matrix of model coefficients for each predictor by each task that are shared between each task
    :param lambda_B: float
        Penalty coefficient for the block matrix
    :return:
    """

    assert cov_C.shape[0] == cov_D.shape[0]
    assert cov_C.shape[1] == cov_D.shape[1]
    assert block_matrix.shape == sparse_matrix.shape
    assert check.argument_type(lambda_B, (float, int))

    n_features = cov_C.shape[1]
    n_tasks = cov_C.shape[0]

    # Cycle through predictors
    for j in range(n_features):

        # Initialize next coefficients
        alphas = np.zeros(n_tasks)
        # Update tasks for each predictor together

        for task_id in range(n_tasks):
            # Task covariance update terms
            task_c = cov_C[task_id]
            task_d = cov_D[task_id]

            # Previous task block-sparse and sparse coefficients
            task_b = block_matrix[:, task_id]
            task_s = sparse_matrix[:, task_id]

            # Set block-sparse coefficient for feature j to zero
            block_tmp = np.copy(task_b)
            block_tmp[j] = 0.

            # Calculate next coefficient based on fit only
            if task_d[j, j] != 0:
                alphas[task_id] = (task_c[j] - np.sum((block_tmp + task_s) * task_d[:, j])) / task_d[j, j]
            else:
                alphas[task_id] = 0.

        # Set all tasks to zero if l1-norm less than lamB
        if np.linalg.norm(alphas, 1) <= lambda_B:
            block_matrix[j, :] = np.zeros(n_tasks)

        # Regularized update for predictors with larger l1-norm
        else:
            # Find number of coefficients that would make l1-norm greater than penalty
            indices = np.abs(alphas).argsort()[::-1]
            sorted_alphas = alphas[indices]
            m_star = np.argmax((np.abs(sorted_alphas).cumsum() - lambda_B) / (np.arange(n_tasks) + 1))

            # Initialize new weights
            new_weights = np.zeros(n_tasks)

            # Keep small coefficients and regularize large ones (in above group)
            for k, idx in enumerate(indices):
                if k > m_star:
                    new_weights[idx] = sorted_alphas[k]
                else:
                    sign = np.sign(sorted_alphas[k])
                    update_term = np.sum(np.abs(sorted_alphas)[:m_star + 1]) - lambda_B
                    new_weights[idx] = (sign / (m_star + 1)) * update_term
            # update current predictor
            block_matrix[j, :] = new_weights

    return block_matrix


def sum_squared_errors(X, Y, betas, task_id):
    """
    Get RSS for a particular task 'k'
    :param X: list(np.ndarray [N x K]) [T]
        List consisting of design matrixes for each task
    :param Y: list(np.ndarray [N x 1]) [T]
        List consisting of response matrixes for each task
    :param betas: np.ndarray [K x T]
        Fit model coefficients for each task
    :param task_id: int
        Task ID
    :return:
    """

    assert check.argument_type(X, list)
    assert check.argument_type(Y, list)
    assert check.argument_type(betas, np.ndarray)
    assert check.argument_type(task_id, int)

    return np.sum((Y[task_id].T - np.dot(X[task_id], betas[:, task_id])) ** 2)


def ebic(X, Y, model_weights, n_tasks, n_samples, n_preds, gamma=1, min_rss=MIN_RSS):
    """
    Calculate EBIC for each task, and take the mean


    Extended Bayesian information criteria for model selection with large model spaces
    Jiahua Chen & Zehua Chen, Biometrika, Volume 95, Issue 3, September 2008, Pages 759–771,

    :param X: list(np.ndarray [N x K]) [T]
        List consisting of design matrixes for each task
    :param Y: list(np.ndarray [N x 1]) [T]
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

        # Find the number of non-zero predictors
        nonzero_pred = (model_weights[:, task_id] != 0).sum()

        # Calculate RSS for the model
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
    :param TFs: list()
        A list of non-zero TFs (k) included in the model
    :param gene: str
        The gene modeled
    :return out_weights: pd.DataFrame [k x 4]
        An edge table (regulator -> target) with the model coefficient and the variance explained by that predictor for
        each non-zero predictor
    """

    assert check.argument_type(X, np.ndarray)
    assert check.argument_type(y, np.ndarray)
    assert check.argument_type(TFs, list)

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
    out_weights.columns = ['regulator', 'target', 'weights', 'resc_weights']

    return out_weights


def _format_prior(priors, gene, tasks, prior_weight):
    """
    Returns weighted priors for one gene
    :param priors: list(pd.DataFrame [G x K]) or pd.DataFrame [G x K]
        Either a list of prior data or a single data frame to use for all tasks
    :param gene: str
        The gene to select from the priors
    :param tasks: list(int)
        A list of task IDs
    :param prior_weight: float, int
        How much to weight the priors
    :return prior_out: np.ndarray [K x T]
        The weighted priors for a specific gene in each task
    """

    assert check.argument_type(priors, (list, pd.DataFrame), allow_none=True)
    assert check.argument_string(gene)
    assert check.argument_type(tasks, list)
    assert check.argument_numeric(prior_weight)

    if priors is None:
        return None

    priors_out = []

    # If the priors are a list, get the gene-specific prior from each task
    if isinstance(priors, list):
        assert len(priors) == len(tasks)
        for k in tasks:
            prior = priors[k].reindex([gene]).replace(np.nan, 0)
            priors_out.append(_weight_prior(prior.loc[gene, :], prior_weight))

    # Otherwise just use the same prior for each task
    else:
        prior = priors.reindex([gene]).replace(np.nan, 0)
        priors_out = [_weight_prior(prior.loc[gene, :], prior_weight)] * len(tasks)

    # Return a [K x T]
    priors_out = np.transpose(np.asarray(priors_out))
    return priors_out


def _weight_prior(prior, prior_weight):
    """
    Weight priors
    :param prior: pd.Series [K] or np.ndarray [K,]
        The prior information
    :param prior_weight: numeric
        How much to weight priors. If this is 1, do not weight prior terms differently
    :return prior: pd.Series [K] or np.ndarray [K,]
        Weighted priors
    """

    # Set non-zero priors to 1 and zeroed priors to 0
    prior = (prior != 0).astype(float)

    # Divide by the prior weight
    prior /= prior_weight

    # Set zeroed priors to 1
    prior[prior == 0] = 1.0

    # Reweight priors
    prior = prior / prior.sum() * len(prior)
    return prior


def _format_weights(df, col, targets, regs):
    """
    Reformat the edge table (target -> regulator) that's output by amusr into a pivoted table that the rest of the
    inferelator workflow can handle
    :param df: pd.DataFrame
        An edge table (regulator -> target) with columns containing model values
    :param col:
        Which column to pivot into values
    :param targets: list
        A list of target genes (the index of the output data)
    :param regs: list
        A list of regulators (the columns of the output data)
    :return out: pd.DataFrame [G x K]
        A [targets x regulators] dataframe pivoted from the edge dataframe
    """

    # Make sure that the value column is all numeric
    df[col] = pd.to_numeric(df[col])

    # Pivot an edge table into a matrix of values
    out = pd.pivot_table(df, index='target', columns='regulator', values=col, fill_value=0.)

    # Reindex to a [targets x regulators] dataframe and fill anything missing with 0s
    out = out.reindex(targets).reindex(regs, axis=1)
    out = out.fillna(value=0.)

    return out


class AMUSRRegressionWorkflow(base_regression.RegressionWorkflow):
    """
    Add AMuSR regression into a workflow object
    """

    def run_regression(self):

        betas = [[] for _ in range(self.n_tasks)]
        rescaled_betas = [[] for _ in range(self.n_tasks)]

        for idx in range(self.num_bootstraps):
            utils.Debug.vprint('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps), level=0)
            current_betas, current_rescaled_betas = self.run_bootstrap(idx)

            if self.is_master():
                for k in range(self.n_tasks):
                    betas[k].append(current_betas[k])
                    rescaled_betas[k].append(current_rescaled_betas[k])

        return betas, rescaled_betas

    def run_bootstrap(self, bootstrap_idx):
        x, y = [], []

        # Select the appropriate bootstrap from each task and stash the data into X and Y
        for k in range(self.n_tasks):
            x.append(self.task_design[k].iloc[:, self.task_bootstraps[k][bootstrap_idx]].transpose())
            y.append(self.task_response[k].iloc[:, self.task_bootstraps[k][bootstrap_idx]].transpose())

        MPControl.sync_processes(pref="amusr_pre")
        regress = AMuSR_regression(x, y, tfs=self.regulators, genes=self.targets, priors=self.priors_data,
                                   prior_weight=self.prior_weight)
        return regress.run()


def filter_genes_on_tasks(list_of_indexes, task_expression_filter):
    """
    Take a list of indexes and filter them based on the method specified in task_expression_filter to a single
    index

    :param list_of_indexes: list(pd.Index)
    :param task_expression_filter: str or int
    :return filtered_genes: pd.Index
    """

    filtered_genes = list_of_indexes[0]

    # If task_expression_filter is a number only keep genes in that number of tasks or higher
    if isinstance(task_expression_filter, int):
        filtered_genes = pd.concat(list(map(lambda x: x.to_series(), list_of_indexes))).value_counts()
        filtered_genes = filtered_genes[filtered_genes >= task_expression_filter].index
    # If task_expression_filter is "intersection" only keep genes in all tasks
    elif task_expression_filter == "intersection":
        for gene_idx in list_of_indexes:
            filtered_genes = filtered_genes.intersection(gene_idx)
    # If task_expression_filter is "union" keep genes that are in any task
    elif task_expression_filter == "union":
        for gene_idx in list_of_indexes:
            filtered_genes = filtered_genes.union(gene_idx)
    else:
        raise ValueError("{v} is not an allowed task_expression_filter value".format(v=task_expression_filter))

    return filtered_genes
