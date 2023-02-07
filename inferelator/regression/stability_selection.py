import math
import warnings
import itertools
import copy
import numpy as np
import pandas.api.types as pat

from sklearn.linear_model import (
    LinearRegression as _LinearRegression,
    Lasso as _Lasso,
    Ridge as _Ridge,
    lasso_path
)

from inferelator.regression.base_regression import (
    BaseRegression,
    PreprocessData,
    _RegressionWorkflowMixin,
    _MultitaskRegressionWorkflowMixin,
    recalculate_betas_from_selected,
    predict_error_reduction,
    gene_data_generator,
    PROGRESS_STR
)

from inferelator.distributed.inferelator_mp import MPControl
from inferelator import utils

_DEFAULT_ALPHAS = np.insert(np.logspace(-2, 1, 20), 0, 0)
_DEFAULT_NUM_SUBSAMPLES = 20
_DEFAULT_THRESHOLD = 0.05
_DEFAULT_SEED = 42
_DEFAULT_METHOD = 'lasso'
_DEFAULT_PARAMS = {"max_iter": 2000}


def _lasso_path(
    x,
    y,
    alphas,
    **kwargs
):

    _coefs = np.zeros((x.shape[1], len(alphas)))
    alphas = np.sort(alphas)[::-1]

    _alpha0 = alphas == 0.

    if np.sum(_alpha0) > 0:
        _coefs[:, _alpha0] = _LinearRegression(
            fit_intercept=False
        ).fit(x, y).coef_[:, None]

    _, _lp_coef, _ = lasso_path(
        x,
        y,
        alphas=alphas[~_alpha0],
        **kwargs
    )

    _coefs[:, ~_alpha0] = _lp_coef
    _coefs = [
        _coefs[:, i].flatten()
        for i in range(_coefs.shape[1])
    ]

    return alphas, _coefs


def _regress_all_alphas(
    x,
    y,
    alphas,
    regression,
    ridge_threshold=1e-2,
    **kwargs
):
    """
    Fit regression with LASSO or Ridge
    on an array of alpha values,
    using OLS for alpha == 0

    Warm start LASSO with the previous
    coefficients.

    :param x: Predictor data
    :type x: np.ndarray
    :param y: Response data
    :type y: np.ndarray
    :param alphas: Regression alphas
    :type alphas: np.ndarray, list
    :param regression: Regression method ('lasso' or 'ridge')
    :type regression: str
    :param ridge_threshold: Shrink values less than this to 0
        for ridge regression, defaults to 1e-2
    :type ridge_threshold: numeric, optional
    :raises ValueError: Raise a ValueError if regression string
        is not 'lasso' or 'ridge'
    :return: A list of model coefficients from regression
    :rtype: list(np.ndarray)
    """

    _regression_coefs = []

    if regression == 'lasso':
        return _lasso_path(
            x,
            y,
            alphas,
            **kwargs
        )
    elif regression == 'ridge':
        pass
    else:
        raise ValueError("regression must be 'lasso' or 'ridge'")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for a in alphas:

            _regression_coefs.append(
                _regress(
                    x,
                    y,
                    a,
                    regression,
                    ridge_threshold=ridge_threshold,
                    **kwargs
                )
            )

    return alphas, _regression_coefs

# Wrapper for a single alpha value
def _regress(
    x,
    y,
    alpha,
    regression,
    ridge_threshold=1e-2,
    **kwargs
):

    if alpha == 0:
        return _LinearRegression(
            fit_intercept=False
        ).fit(x, y).coef_.copy()
    elif regression == 'lasso':
        return _Lasso(
            alpha=alpha,
            fit_intercept=False,
            **kwargs
        ).fit(x, y).coef_.copy()
    elif regression == 'ridge':
        _coefs = _Ridge(
            alpha=alpha,
            fit_intercept=False,
            **kwargs
        ).fit(x, y).coef_.copy()

        _coefs[_coefs < ridge_threshold] = 0.

        return _coefs
    else:
        raise ValueError(
            "regression must be 'lasso' or 'ridge'"
        )

def stars_model_select(
    x,
    y,
    alphas,
    threshold=_DEFAULT_THRESHOLD,
    num_subsamples=_DEFAULT_NUM_SUBSAMPLES,
    random_seed=_DEFAULT_SEED,
    method='lasso',
    **kwargs
):
    """
    Model using StARS (Stability Approach to Regularization Selection) for model selection

    :param x:
    :param y:
    :param alphas:
    :param threshold:
    :param num_subsamples:
    :param random_seed:
    :param method:
    :param kwargs:
    :return:
    """

    # Number of obs
    n, k = x.shape

    # Sort alphas
    alphas = np.sort(alphas)[::-1]

    if n < num_subsamples:
        raise ValueError(
            f"Subsamples ({num_subsamples}) for StARS is larger "
            f"than the number of samples ({n})"
        )

    # Calculate the number of obs per subsample
    b = math.floor(n / num_subsamples)

    # Make an index for subsampling
    idx = _make_subsample_idx(n, b, num_subsamples, random_seed=random_seed)

    # Calculate betas for stability selection
    betas = {a: [] for a in alphas}

    for sample in range(num_subsamples):
        # Sample and put into column-major (the coordinate descent
        # implementation in sklearn wants that order)
        x_samp = np.asarray(x[idx == sample, :], order='F')
        y_samp = y[idx == sample]

        _beta_alphas, _beta_coefs = _regress_all_alphas(
            x_samp,
            y_samp,
            alphas,
            method,
            **kwargs
        )

        for _coef, a in zip(_beta_coefs, _beta_alphas):
            betas[a].append(_coef)

    # Calculate edge stability
    stabilities = {
        a: _calculate_stability(betas[a])
        for a in alphas
    }

    # Calculate monotonic increasing (as alpha decreases) mean edge stability
    total_instability = np.maximum.accumulate(
        [np.mean(stabilities[a]) for a in alphas]
    )

    threshold_alphas = np.array(alphas)[total_instability < threshold]
    selected_alpha = np.min(threshold_alphas) if len(threshold_alphas) > 0 else alphas[0]

    refit_betas = _regress(
        x,
        y,
        selected_alpha,
        method,
        **kwargs
    )

    beta_nonzero = _make_bool_matrix(refit_betas)

    if beta_nonzero.sum() == 0:
        return dict(pp=np.repeat(True, k).tolist(),
                    betas=np.zeros(k),
                    betas_resc=np.zeros(k))
    else:
        x = x[:, beta_nonzero]
        utils.make_array_2d(y)
        betas = recalculate_betas_from_selected(x, y)
        betas_resc = predict_error_reduction(x, y, betas)

        return dict(pp=beta_nonzero,
                    betas=betas,
                    betas_resc=betas_resc)


def _calculate_stability(edges):
    """

    :param edges:
    :return:
    """

    edge_sum = np.zeros(edges[0].shape, dtype=int)
    for e in edges:
        edge_sum += _make_bool_matrix(e)

    edge_sum = edge_sum / len(edges)
    return 2 * edge_sum * (1 - edge_sum)


def _make_bool_matrix(edge_matrix):

    if pat.is_float_dtype(edge_matrix.dtype):
        _eps = np.finfo(dtype=edge_matrix.dtype).eps
        return np.abs(edge_matrix) > _eps
    else:
        return edge_matrix != 0


def _make_subsample_idx(n, b, num_subsamples, random_seed=42):
    # Partition all samples into num_subsamples groups
    subsample_index = np.zeros((n,), dtype=np.int8)
    for i in range(b):
        start, stop = i * num_subsamples, min((i + 1) * num_subsamples, n)
        subsample_index[start:stop] = list(range(num_subsamples))[0:stop - start]

    np.random.RandomState(random_seed).shuffle(subsample_index)

    return subsample_index


class StARS(BaseRegression):

    def __init__(
        self,
        X,
        Y,
        random_seed,
        alphas=_DEFAULT_ALPHAS,
        num_subsamples=_DEFAULT_NUM_SUBSAMPLES,
        method=_DEFAULT_METHOD,
        parameters=None
    ):

        self.random_seed = random_seed
        self.alphas = alphas
        self.num_subsamples = num_subsamples
        self.method = method

        self.params = parameters if parameters is not None else {}

        super(StARS, self).__init__(X, Y)

    def regress(self):
        """
        Execute StARS

        :return: list
            Returns a list of regression results that base_regression
            pileup_data can process
        """

        x = self.X.values
        nG = self.G

        return MPControl.map(
            _stars_regression_wrapper,
            itertools.repeat(x, nG),
            gene_data_generator(self.Y, nG),
            itertools.repeat(self.alphas, nG),
            range(nG),
            self.genes,
            itertools.repeat(nG, nG),
            method=self.method,
            num_subsamples=self.num_subsamples,
            random_seed=self.random_seed,
            **self.params,
            scatter=[x]
        )


def _stars_regression_wrapper(x, y, alphas, j, gene, nG, **kwargs):

    utils.Debug.vprint(
        PROGRESS_STR.format(gn=gene, i=j, total=nG),
        level=0 if j % 1000 == 0 else 2
    )

    data = stars_model_select(
        x,
        PreprocessData.preprocess_response_vector(y),
        alphas,
        **kwargs
    )

    data['ind'] = j
    return data


class StARSWorkflowMixin(_RegressionWorkflowMixin):
    """
    Stability Approach to Regularization Selection (StARS)-LASSO.
    StARS-Ridge is implemented on an experimental basis.

    https://arxiv.org/abs/1006.3316
    https://doi.org/10.1016/j.immuni.2019.06.001
    """

    sklearn_params = copy.copy(_DEFAULT_PARAMS)
    alphas = _DEFAULT_ALPHAS
    regress_method = _DEFAULT_METHOD
    num_subsamples = _DEFAULT_NUM_SUBSAMPLES

    def __init__(self, *args, **kwargs):
        self.sklearn_params = {}
        super(StARSWorkflowMixin, self).__init__(*args, **kwargs)

    def set_regression_parameters(
        self,
        alphas=None,
        num_subsamples=None,
        method=None,
        **kwargs
    ):
        """
        Set regression parameters for StARS-LASSO

        :param alphas: A list of alpha (L1 term) values to search.
            Defaults to logspace between 0. and 10.
        :type alphas: list(float)
        :param num_subsamples: The number of groups to break data
            into. Defaults to 20.
        :type num_subsamples: int
        :param method: The model to use. Can choose from 'lasso'
            or 'ridge'. Defaults to 'lasso'.
            If 'ridge' is set, ridge_threshold should also be passed.
            Any value below ridge_threshold will be set to 0.
        :type method: str
        :param kwargs: Any additional arguments will be passed to the
            LASSO or Ridge scikit-learn object at instantiation
        :type kwargs: any
        """

        self.sklearn_params.update(kwargs)

        self._set_with_warning(
            'alphas',
            alphas
        )

        self._set_with_warning(
            'num_subsamples',
            num_subsamples
        )

        self._set_with_warning(
            'regress_method',
            method
        )

    def run_regression(self):

        betas, resc_betas = StARS(
            self.design,
            self.response,
            self.random_seed,
            alphas=self.alphas,
            method=self.regress_method,
            num_subsamples=self.num_subsamples,
            parameters=self.sklearn_params
        ).run()

        return [betas], [resc_betas], betas, resc_betas


class StARSWorkflowByTaskMixin(
    _MultitaskRegressionWorkflowMixin,
    StARSWorkflowMixin
):
    """
    Stability Approach to Regularization Selection (StARS)-LASSO.
    StARS-Ridge is implemented on an experimental basis.

    https://arxiv.org/abs/1006.3316
    https://doi.org/10.1016/j.immuni.2019.06.001
    """

    def run_regression(self):
        betas, betas_resc = [], []

        # Run tasks individually
        for k in range(self._n_tasks):

            utils.Debug.vprint(
                f'Calculating task {k} betas using StARS',
                level=0
            )

            t_beta, t_br = StARS(
                self._task_design[k],
                self._task_response[k],
                self.random_seed,
                alphas=self.alphas,
                method=self.regress_method,
                num_subsamples=self.num_subsamples,
                parameters=self.sklearn_params
            ).run()

            betas.append([t_beta])
            betas_resc.append([t_br])

        _unpack_betas = [x[0] for x in betas]
        _unpack_var_exp = [x[0] for x in betas_resc]

        return betas, betas_resc, _unpack_betas, _unpack_var_exp
