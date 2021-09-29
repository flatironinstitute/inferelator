import numpy as np
import pandas as pd
import os

from scipy.special import comb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from inferelator.distributed.inferelator_mp import MPControl
from inferelator import utils
from inferelator.utils import Validator as check
from inferelator import default
from inferelator.regression import base_regression

DEFAULT_prior_weight = 1.0
DEFAULT_Cs = np.logspace(np.log10(0.01), np.log10(10), 20)[::-1]

MAX_ITER = 1000
TOL = 1e-2
REL_TOL = None
MIN_WEIGHT_VAL = 0.1
MIN_RSS = 1e-10


def run_regression_EBIC(X, Y, TFs, tasks, gene, prior, Cs=None, Ss=None, lambda_Bs=None,
                        lambda_Ss=None, scale_data=False, return_lambdas=False, tol=TOL, rel_tol=REL_TOL,
                        use_numba=False):
    """
    Run multitask regression. Search the regularization coefficient space and select the model with the
    lowest eBIC.

    :param X: list(np.ndarray [N x K]) [t]
        List consisting of design matrixes for each task
    :param Y: list(np.ndarray [N x 1]) [t]
        List consisting of response matrixes for each task
    :param TFs: list [K]
        List of TF names for each feature
    :param tasks: list(int) [t]
        List identifying each task
    :param gene: str
        The gene being modeled
    :param prior: np.ndarray [K x T]
        The priors for this gene in a TF x Task array
    :param scale_data:
    :param return_lambdas:
    :param tol:
    :param rel_tol:
    :param use_numba: bool
    :return: dict
    """

    if use_numba:
        AMuSR_math.set_numba()

    assert len(X) == len(Y)
    assert len(X) == len(tasks)
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
            combined_weights, sparse_matrix, block_matrix = amusr_fit(cov_C, cov_D, b, s,
                                                                      sparse_matrix, block_matrix, prior,
                                                                      tol=tol, rel_tol=rel_tol)

            # Score model
            ebic_score = ebic(X, Y, combined_weights, n_tasks, n_samples, n_preds)

            # Keep the model if it's the lowest scoring
            if ebic_score < min_ebic:
                min_ebic = ebic_score
                model_output = combined_weights
                opt_b, opt_s = b, s

    ###### RESCALE WEIGHTS ######
    output = {}

    if model_output is not None:
        for kx, k in enumerate(tasks):
            nonzero = model_output[:,kx] != 0
            if nonzero.sum() > 0:
                cTFs = np.asarray(TFs)[model_output[:,kx] != 0]
                output[k] = _final_weights(X[kx][:, nonzero], Y[kx], cTFs, gene)

    return (output, opt_b, opt_s) if return_lambdas else output

class AMuSR_regression(base_regression.BaseRegression):

    X = None  # list(InferelatorData)
    Y = None  # list(InferelatorData)
    priors = None  # list(pd.DataFrame [G, K]) OR pd.DataFrame [G, K]
    tfs = None  # pd.Index OR list
    genes = None  # pd.Index OR list

    K = None  # int
    G = None  # int
    n_tasks = None  # int
    prior_weight = DEFAULT_prior_weight  # float
    remove_autoregulation = True  # bool

    lambda_Bs = None
    lambda_Ss = None
    Cs = None
    Ss = None

    tol = None
    rel_tol = None
    
    use_numba = False

    regression_function = staticmethod(run_regression_EBIC)

    def __init__(self, X, Y, tfs=None, genes=None, priors=None, prior_weight=1, remove_autoregulation=True,
                 lambda_Bs=None, lambda_Ss=None, Cs=None, Ss=None, tol=TOL, rel_tol=REL_TOL, use_numba=False):
        """
        Set up a regression object for multitask regression
        :param X: list(InferelatorData)
        :param Y: list(InferelatorData)
        :param priors: pd.DataFrame [G, K]
        :param prior_weight: float
        :param remove_autoregulation: bool
        :param lambda_Bs: list(float) [None]
        :param lambda_Ss: list(float) [None]
        :param Cs: list(float) [None] 
        :param Ss: list(float) [None]
        :param use_numba: bool [False]
        """

        # Check input types are correct
        assert check.argument_type(X, list)
        assert check.argument_type(Y, list)
        assert check.argument_type(tfs, (list, pd.Series, pd.Index), allow_none=True)
        assert check.argument_type(genes, (list, pd.Series, pd.Index), allow_none=True)
        assert check.argument_numeric(prior_weight)
        assert check.argument_numeric(tol)
        assert check.argument_numeric(rel_tol, allow_none=True)
        assert len(X) == len(Y)

        # Set the data into the regression object
        self.X = scale_list_of_data(X)
        self.Y = scale_list_of_data(Y)
        self.n_tasks = len(X)

        # Set the priors and weight into the regression object
        self.priors = priors
        self.prior_weight = float(prior_weight)

        # Construct a list of regulators if they are not passed in from the union of the task regulators
        if tfs is None:
            tfs = [design.gene_names for design in X]
            self.tfs = filter_genes_on_tasks(tfs, "union")
        else:
            self.tfs = tfs

        # Construct a list of genes if they are not passed in from the union of the task genes
        if genes is None:
            genes = [resp.gene_names for resp in Y]
            self.genes = filter_genes_on_tasks(genes, "union")
        else:
            self.genes = genes

        # Set the regulators and targets into the regression object
        self.K, self.G = len(tfs), len(genes)
        self.remove_autoregulation = remove_autoregulation

        # Set the regularization coefficients into the regression object
        self.lambda_Bs = lambda_Bs
        self.lambda_Ss = lambda_Ss
        self.Cs = Cs
        self.Ss = Ss

        # Set the tolerances into the regression object
        self.tol = tol
        self.rel_tol = rel_tol

        self.use_numba = use_numba

    def regress(self, regression_function=None):
        """
        Execute multitask (AMUSR)
        :return: list
            Returns a list of regression results that the amusr_regression pileup_data can process
        """

        regression_function = self.regression_function if regression_function is None else regression_function

        if MPControl.is_dask():
            from inferelator.distributed.dask_functions import amusr_regress_dask
            return amusr_regress_dask(self.X, self.Y, self.priors, self.prior_weight, self.n_tasks, self.genes,
                                      self.tfs, self.G, remove_autoregulation=self.remove_autoregulation,
                                      regression_function=regression_function,
                                      tol=self.tol, rel_tol=self.rel_tol, use_numba=self.use_numba)

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
                if gene in self.Y[k].gene_names:
                    x.append(self.X[k].get_gene_data(tfs))  # list([N, K])
                    y.append(self.Y[k].get_gene_data(gene, force_dense=True).reshape(-1, 1))  # list([N, 1])
                    tasks.append(k)  # [T,]

            prior = format_prior(self.priors, gene, tasks, self.prior_weight, tfs=tfs)
            return regression_function(x, y, tfs, tasks, gene, prior, Cs=self.Cs, Ss=self.Ss,
                                       lambda_Bs=self.lambda_Bs, lambda_Ss=self.lambda_Ss, 
                                       tol=self.tol, rel_tol=self.rel_tol, use_numba=self.use_numba)

        return MPControl.map(regression_maker, range(self.G))

    def pileup_data(self, run_data):

        weights = []
        rescaled_weights = []

        for k in range(self.n_tasks):
            results_k = []
            for res in run_data:
                try:
                    results_k.append(res[k])
                except (KeyError, IndexError):
                    pass

            results_k = pd.concat(results_k)
            weights_k = _format_weights(results_k, 'weights', self.genes, self.tfs)
            rescaled_weights_k = _format_weights(results_k, 'resc_weights', self.genes, self.tfs)
            rescaled_weights_k[rescaled_weights_k < 0.] = 0

            weights.append(weights_k)
            rescaled_weights.append(rescaled_weights_k)

        return weights, rescaled_weights

def amusr_fit(cov_C, cov_D, lambda_B=0., lambda_S=0., sparse_matrix=None, block_matrix=None, prior=None,
              max_iter=MAX_ITER, tol=TOL, rel_tol=REL_TOL, rel_tol_min_iter=10, min_weight=MIN_WEIGHT_VAL):
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
        sparse_matrix = AMuSR_math.updateS(cov_C, cov_D, block_matrix, sparse_matrix,
                                           lambda_S, prior, n_tasks, n_features)

        block_matrix = AMuSR_math.updateB(cov_C, cov_D, block_matrix, sparse_matrix,
                                          lambda_B, prior, n_tasks, n_features)

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
    out_weights.columns = ['regulator', 'target', 'weights', 'resc_weights']

    return out_weights


def format_prior(priors, gene, tasks, prior_weight, tfs=None):
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

    def _reindex_to_gene(p):
            p = p.reindex([gene])
            p = p.reindex(tfs, axis=1) if tfs is not None else p
            p = p.fillna(0.0)
            return p

    # If the priors are a list, get the gene-specific prior from each task
    if isinstance(priors, list) and len(priors) > 1:
               
        priors_out = [_weight_prior(_reindex_to_gene(priors[k]).loc[gene, :].values, prior_weight) for k in tasks]
        priors_out = np.transpose(np.vstack(priors_out))

    # Otherwise just use the same prior for each task
    else:

        priors = priors[0] if isinstance(priors, list) else priors
        priors_out = np.tile(_weight_prior(_reindex_to_gene(priors).loc[gene, :].values, prior_weight).reshape(-1, 1),
                             (1, len(tasks)))

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

def scale_list_of_arrays(X):
    """
    Scale a list of arrays so that each has mean 0 and unit variance
    
    :param X: list(np.ndarray) [T]
    :return X: list(np.ndarray) [T]
    """

    assert check.argument_type(X, list)

    return [StandardScaler().fit_transform(xk.astype(float)) for xk in X]

def scale_list_of_data(X):
    """
    Scale a list of data objects so that each has mean 0 and unit variance

    :param X: list(InferelatorData) [T]
    :return X: list(InferelatorData) [T]
    """

    assert check.argument_type(X, list)

    return [xk.zscore(ddof=0) for xk in X]

class AMUSRRegressionWorkflowMixin(base_regression._MultitaskRegressionWorkflowMixin):
    """
    Multi-Task AMuSR regression

    https://doi.org/10.1371/journal.pcbi.1006591
    """

    prior_weight = default.DEFAULT_prior_weight

    # Model hyperparameters
    lambda_Bs = None
    lambda_Ss = None
    heuristic_Cs = None

    tol = TOL
    relative_tol = REL_TOL
  
    def set_regression_parameters(self, prior_weight=None, lambda_Bs=None, lambda_Ss=None, heuristic_Cs=None,
                                  tol=None, relative_tol=None, use_numba=None):
        """
        Set regression parameters for AmUSR.

        :param prior_weight: Weight for edges that are present in the prior network.
            Non-prior edges have a weight of 1. Set this to 1 to weight prior and non-prior edges equally.
            Defaults to 1.
        :type prior_weight: numeric
        :param lambda_Bs: Lambda_B values to search during model selection.
            If not set, lambda_B will be chosen using the heuristic lambda_b = c * sqrt(d log p / n) from Castro 2019
            Defaults to not set. Must be provided if lambda_S is set.
        :type lambda_Bs: list(floats) or np.ndarray(floats)
        :param lambda_Ss: Lambda_S values to search during model selection.
            If not set, lambda_S will be chosen using the heuristic 0.5 < lambda_s/lambda_b < 1 from Castro 2019
            Defaults to not set.
        :type lambda_Ss: list(floats) or np.ndarray(floats)
        :param heuristic_Cs: c values to search during model selection.
            Values of c to calculate lambda_b = c * sqrt(d log p / n),
            Defaults to np.logspace(np.log10(0.01), np.log10(10), 20)[::-1].
            Does not have an effect if lambda_B is provided.
        :type heuristic_Cs: list(floats) or np.ndarray(floats)
        :param tol: Convergence tolerance for amusr regression
        :type tol: float
        :param relative_tol: Relative convergence tolerance for amusr regression
        :type relative_tol: float
        """

        self._set_with_warning("prior_weight", prior_weight)
        self._set_with_warning("lambda_Bs", lambda_Bs)
        self._set_with_warning("lambda_Ss", lambda_Ss)
        self._set_with_warning("heuristic_Cs", heuristic_Cs)
        self._set_without_warning("tol", tol)
        self._set_without_warning("relative_tol", relative_tol)


    def run_bootstrap(self, bootstrap_idx):
        x, y = [], []

        # Select the appropriate bootstrap from each task and stash the data into X and Y
        for k in range(self._n_tasks):
            x.append(self._task_design[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx]))
            y.append(self._task_response[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx]))

        regress = AMuSR_regression(x, y, tfs=self._regulators, genes=self._targets, priors=self._task_priors,
                                   prior_weight=self.prior_weight, lambda_Bs=self.lambda_Bs, lambda_Ss=self.lambda_Ss, 
                                   Cs=self.heuristic_Cs, tol=self.tol, rel_tol=self.relative_tol, 
                                   use_numba=self.use_numba)
                                   
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
                B[j,:] = np.zeros(n_tasks)

            # regularized update for predictors with larger l1-norm
            else:
                # find number of coefficients that would make l1-norm greater than penalty
                indices = np.abs(alphas).argsort()[::-1]
                sorted_alphas = alphas[indices]
                m_star = np.argmax((np.abs(sorted_alphas).cumsum()-lamB)/(np.arange(n_tasks)+1))
                # initialize new weights
                new_weights = np.zeros(n_tasks)
                # keep small coefficients and regularize large ones (in above group)
                for k in range(n_tasks):
                    idx = indices[k]
                    if k > m_star:
                        new_weights[idx] = sorted_alphas[k]
                    else:
                        sign = np.sign(sorted_alphas[k])
                        update_term = np.sum(np.abs(sorted_alphas)[:m_star+1])-lamB
                        new_weights[idx] = (sign/(m_star+1))*update_term
                # update current predictor
                B[j,:] = new_weights

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
