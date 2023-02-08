import numpy as np
import pandas as pd
import scipy.stats

from inferelator.utils import (
    Debug,
    Validator as check
)

from inferelator.preprocessing.data_normalization import PreprocessData

DEFAULT_CHUNK = 25
PROGRESS_STR = "Regression on {gn} [{i} / {total}]"


class BaseRegression:
    # These are all the things that have to be set in a new regression class

    chunk = DEFAULT_CHUNK  # int

    # Raw Data
    X = None  # [K x N] float
    Y = None  # [G x N] float
    G = None  # int G
    K = None  # int K

    def __init__(self, X, Y):
        """
        Create a regression object and do basic data transforms

        :param X: Expression or Activity data [N x K]
        :type X: InferelatorData
        :param Y: Response expression data [N x G]
        :type Y: InferelatorData
        """

        # Get the IDs and total count for the genes and predictors
        self.K = X.num_genes
        self.tfs = X.gene_names
        self.G = Y.num_genes
        self.genes = Y.gene_names

        # Rescale the design expression or activity data on features
        self.X = PreprocessData.preprocess_design(X)
        self.Y = Y

        Debug.vprint(
            f"Predictor matrix {X.shape} and response matrix {Y.shape} ready"
        )

    def run(self):
        """
        Execute regression separately on each response variable in the data

        :return: pd.DataFrame [G x K], pd.DataFrame [G x K]
            Returns the regression betas and beta error reductions for all
            threads if this is the master thread (rank 0)
            Returns None, None if it's a subordinate thread
        """

        return self.pileup_data(self.regress())

    def regress(self):
        """
        Execute regression and return a list which can be provided to
        pileup_data
        :return: list
        """
        raise NotImplementedError

    def pileup_data(self, run_data):
        """
        Take the completed run data and pack it up into a DataFrame of betas

        :param run_data: A list of regression result dicts ordered by gene.
            Each regression result should have `ind`, `pp`, `betas` and
            `betas_resc` keys with the appropriate data.
        :type: run_data: list
        :return betas, betas_rescale: [G x K] DataFrames
        :rtype: pd.DataFrame, pd.DataFrame
        """

        # Create G x K arrays of 0s to populate with the regression data
        betas = np.zeros((self.G, self.K), dtype=np.dtype(float))
        betas_rescale = np.zeros((self.G, self.K), dtype=np.dtype(float))

        # Populate the zero arrays with the BBSR betas
        for data in run_data:

            # If data is None assume a null model
            if data is None:
                raise RuntimeError("No model produced by regression method")

            xidx = data['ind']  # Int
            yidx = data['pp']  # Boolean array of size K
            betas[xidx, yidx] = data['betas']
            betas_rescale[xidx, yidx] = data['betas_resc']

        d_len, b_avg, null_m = self._summary_stats(betas)

        Debug.vprint(
            "Regression complete: "
            f"{d_len} Models, {b_avg:.02f} Preds per Model ({null_m} Null)",
            level=0
        )

        # Convert arrays into pd.DataFrames to return results
        betas = pd.DataFrame(
            betas,
            index=self.Y.gene_names,
            columns=self.X.gene_names
        )

        betas_rescale = pd.DataFrame(
            betas_rescale,
            index=self.Y.gene_names,
            columns=self.X.gene_names
        )

        return betas, betas_rescale

    @staticmethod
    def _summary_stats(arr):
        d_len = arr.shape[0]
        b_avg = np.mean(np.sum(arr != 0, axis=1))
        null_m = np.sum(np.sum(arr != 0, axis=1) == 0)
        return d_len, b_avg, null_m


class _RegressionWorkflowMixin(object):
    """
    RegressionWorkflow implements run_regression and run_bootstrap
    Each regression method needs to extend this to implement
    run_bootstrap (and also run_regression if necessary)
    """

    def set_regression_parameters(self, **kwargs):
        """
        Set any parameters which are specific to one or another
        regression method
        """
        pass

    def run_regression(self):
        betas = []
        rescaled_betas = []

        for idx, bootstrap in enumerate(self.get_bootstraps()):

            Debug.vprint(
                f'Bootstrap {idx + 1} of {self.num_bootstraps}',
                level=0
            )

            np.random.seed(self.random_seed + idx)

            current_betas, current_rescaled_betas = self.run_bootstrap(bootstrap)

            betas.append(current_betas)
            rescaled_betas.append(current_rescaled_betas)

        Debug.vprint(
            'Fitting final full model',
            level=0
        )

        full_betas, full_rescaled = self.run_bootstrap(None)

        return betas, rescaled_betas, full_betas, full_rescaled

    def run_bootstrap(self, bootstrap):
        raise NotImplementedError


class _MultitaskRegressionWorkflowMixin(_RegressionWorkflowMixin):
    """
    MultitaskRegressionWorkflow implements run_regression and
    run_bootstrap for multitask workflow

    Each regression method needs to extend this to implement
    run_bootstrap (and also run_regression if necessary)
    """

    def run_regression(self):

        betas = [[] for _ in range(self._n_tasks)]
        rescaled_betas = [[] for _ in range(self._n_tasks)]

        for idx in range(self.num_bootstraps):
            Debug.vprint(
                f'Bootstrap {idx + 1} of {self.num_bootstraps}',
                level=0
            )

            current_betas, current_rescaled_betas = self.run_bootstrap(idx)

            for k in range(self._n_tasks):
                betas[k].append(current_betas[k])
                rescaled_betas[k].append(current_rescaled_betas[k])

        Debug.vprint(
            'Fitting final full model',
            level=0
        )

        full_betas, full_rescaled = self.run_bootstrap(None)

        return betas, rescaled_betas, full_betas, full_rescaled

    def run_bootstrap(self, bootstrap):
        raise NotImplementedError


def recalculate_betas_from_selected(x, y, idx=None):
    """
    Estimate betas from a selected subset of predictors
    :param x: np.ndarray [N x k]
        Predictor matrix
    :param y: np.ndarray [N x 1]
        Response matrix
    :param idx: np.ndarray [k x 1]
        Predictors to use (unused predictors will return a beta of 0)
        If None, use all predictors
    :return: np.ndarray [k,]
        Estimated beta-hats
    """

    # Create an array of size [k,] to hold the estimated betas
    best_betas = np.zeros(x.shape[1], dtype=np.dtype(float))

    # Use all predictors if no subset index is passed in
    if idx is None:
        idx = np.ones(x.shape[1], dtype=np.dtype(bool))

    # Convert boolean array to an array of indexes
    idx = bool_to_index(idx)

    # Subset the predictors with the index array
    x = x[:, idx]

    # Solve for beta-hat with LAPACK or return a null model if xTx is singular
    xtx = np.dot(x.T, x)
    if np.linalg.matrix_rank(xtx) == xtx.shape[1]:
        beta_hat = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
    else:
        beta_hat = np.zeros(len(idx), dtype=np.dtype(float))

    # Use the index array to write beta-hats
    # This yields the same size result matrix as number of predictors in x
    # (even if x is subset with an index)
    for i, j in enumerate(idx):
        best_betas[j] = beta_hat[i]
    return best_betas


def predict_error_reduction(x, y, betas):
    """
    Predict the error reduction from each predictor
    :param x: np.ndarray [n x k]
    :param y: np.ndarray [n x 1]
    :param betas: np.ndarray [k x 1]
    :return: np.ndarray [k,]
    """
    assert check.argument_type(betas, np.ndarray)

    (n, k) = x.shape
    pp_idx = index_of_nonzeros(betas).tolist()

    # Calculate the variance of the residuals
    ss_all = sigma_squared(x, y, betas)
    error_reduction = np.zeros(k, dtype=np.dtype(float))

    # If there's only one predictor, use the ratio of explained variance
    # to total variance (a model with zero predictors)
    if len(pp_idx) == 1:
        error_reduction[pp_idx] = 1 - (ss_all / np.var(y, ddof=1))
        return error_reduction

    for pp_i in range(len(pp_idx)):
        # Copy the index of predictors
        leave_out = pp_idx.copy()
        # Pull off one of the predictors
        lost = leave_out.pop(pp_i)

        # Reestimate betas for all the predictors except the one
        # that we removed
        x_leaveout = x[:, leave_out]

        # Do a standard solve holding out one of the predictors
        try:
            xt = x_leaveout.T
            beta_hat = scipy.linalg.solve(
                np.dot(xt, x_leaveout),
                np.dot(xt, y),
                assume_a='sym'
            )

        # But if it fails use all zero coefficients
        # it shouldn't fail at this stage though
        except np.linalg.LinAlgError:
            beta_hat = np.zeros(len(leave_out), dtype=np.dtype(float))

        # Calculate the variance of the residuals for the new estimated betas
        ss_leaveout = sigma_squared(x_leaveout, y, beta_hat)

        # Check to make sure that the ss_all and ss_leaveout differences
        # aren't just precision-related
        _eps = np.finfo(float).eps * len(pp_idx)
        if np.abs(ss_all - ss_leaveout) < _eps:
            error_reduction[lost] = 0.
        elif ss_leaveout < _eps:
            error_reduction[lost] = 1.
        else:
            error_reduction[lost] = 1 - (ss_all / ss_leaveout)

    return error_reduction


def sigma_squared(x, y, betas):
    return np.var(
        np.subtract(
            y,
            np.dot(x, betas).reshape(-1, 1)
        ),
        ddof=1
    )


def index_of_nonzeros(arr):
    """
    Returns an array that indexes all the non-zero elements of an array
    :param arr: np.ndarray
    :return: np.ndarray
    """
    return np.where(arr != 0)[0]


def bool_to_index(arr):
    """
    Returns an array that indexes all the True elements of a boolean array
    :param arr: np.ndarray
    :return: np.ndarray
    """
    assert check.argument_type(arr, np.ndarray)
    return np.where(arr)[0]


def gene_data_generator(Y, nG):
    """
    Generator for slicing out individual gene records
    And then centering and scaling them

    :param Y: Gene expression data
    :type Y: InferelatorData
    :param nG: Total number of genes to model
    :type nG: int
    :yield: Sliced data
    :rtype: np.ndarray
    """

    for j in range(nG):
        yield Y.get_gene_data(
            j,
            force_dense=True,
            flatten=True
        )
