import math
import warnings
import numpy as np
import pandas.api.types as pat
from sklearn.linear_model import LinearRegression as _LinearRegression, Lasso as _Lasso

from inferelator.regression import base_regression
from inferelator.distributed.inferelator_mp import MPControl
from inferelator.utils import Validator as check
from inferelator import utils

_DEFAULT_ALPHAS = np.insert(np.logspace(-2, 1, 20), 0, 0)
_DEFAULT_NUM_SUBSAMPLES = 20
_DEFAULT_THRESHOLD = 0.05
_DEFAULT_SEED = 42
_DEFAULT_PARAMS = {"max_iter": 2000}


def lasso(x, y, alpha, **kwargs):
    if alpha == 0:
        return _LinearRegression(normalize=False).fit(x, y).coef_
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _Lasso(alpha=alpha, normalize=False, **kwargs).fit(x, y).coef_


def stars_model_select(x, y, alphas, threshold=_DEFAULT_THRESHOLD, num_subsamples=_DEFAULT_NUM_SUBSAMPLES,
                       random_seed=_DEFAULT_SEED, **kwargs):
    """
    Model using StARS (Stability Approach to Regularization Selection) for model selection

    :param x:
    :param y:
    :param alphas:
    :param threshold:
    :param num_subsamples:
    :param random_seed:
    :param kwargs:
    :return:
    """
    # Number of obs
    n, k = x.shape

    if n < num_subsamples:
        msg = "Subsamples ({ns}) for StARS is larger than the number of samples ({n})".format(ns=num_subsamples, n=n)
        raise ValueError(msg)

    # Calculate the number of obs per subsample
    b = math.floor(n / num_subsamples)

    # Make an index for subsampling
    idx = _make_subsample_idx(n, b, num_subsamples, random_seed=random_seed)

    # Calculate betas for stability selection
    betas = {a: [] for a in alphas}
    for sample in range(num_subsamples):
        # Sample and put into column-major (the coordinate descent implementation in sklearn wants that order)
        x_samp = np.asarray(x[idx == sample, :], order='F')
        y_samp = y[idx == sample]

        for a in alphas:
            betas[a].append(lasso(x_samp, y_samp, a, **kwargs))

    # Calculate edge stability
    stabilities = {a: _calculate_stability(betas[a]) for a in alphas}

    # Calculate monotonic increasing (as alpha decreases) mean edge stability
    alphas = np.sort(alphas)[::-1]
    total_instability = np.array([np.mean(stabilities[a]) for a in alphas])

    for i in range(1, len(total_instability)):
        if total_instability[i] < total_instability[i - 1]:
            total_instability[i] = total_instability[i - 1]

    selected_alpha = np.min(np.array(alphas)[total_instability < threshold])

    refit_betas = lasso(x, y, selected_alpha, **kwargs)
    beta_nonzero = _make_bool_matrix(refit_betas)

    if beta_nonzero.sum() == 0:
        return dict(pp=np.repeat(True, k).tolist(),
                    betas=np.zeros(k),
                    betas_resc=np.zeros(k))
    else:
        x = x[:, beta_nonzero]
        utils.make_array_2d(y)
        betas = base_regression.recalculate_betas_from_selected(x, y)
        betas_resc = base_regression.predict_error_reduction(x, y, betas)

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
        return np.abs(edge_matrix) > np.finfo(dtype=edge_matrix.dtype).eps
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


class StARS(base_regression.BaseRegression):

    def __init__(self, X, Y, random_seed, alphas=_DEFAULT_ALPHAS, num_subsamples=_DEFAULT_NUM_SUBSAMPLES, parameters=None):
        self.random_seed = random_seed
        self.alphas = alphas
        self.num_subsamples = num_subsamples

        self.params = parameters if parameters is not None else {}

        super(StARS, self).__init__(X, Y)

    def regress(self):
        """
        Execute Elastic Net

        :return: list
            Returns a list of regression results that base_regression's pileup_data can process
        """

        if MPControl.is_dask():
            from inferelator.distributed.dask_functions import lasso_stars_regress_dask
            return lasso_stars_regress_dask(self.X, self.Y, self.alphas, self.num_subsamples, self.random_seed,
                                            self.params, self.G, self.genes)

        def regression_maker(j):
            level = 0 if j % 100 == 0 else 2
            utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=self.genes[j], i=j, total=self.G), level=level)

            data = stars_model_select(self.X.values,
                                      utils.scale_vector(self.Y.get_gene_data(j, force_dense=True).flatten()),
                                      self.alphas,
                                      num_subsamples=self.num_subsamples,
                                      random_seed=self.random_seed,
                                      **self.params)
            data['ind'] = j
            return data

        return MPControl.map(regression_maker, range(self.G), tell_children=False)


class StARSWorkflow(base_regression.RegressionWorkflow):
    """
    Add elasticnet regression into a workflow object
    """

    sklearn_params = _DEFAULT_PARAMS
    alphas = _DEFAULT_ALPHAS
    num_subsamples = _DEFAULT_NUM_SUBSAMPLES

    def set_regression_parameters(self, alphas=None, num_subsamples=20, **kwargs):
        """
        Set regression parameters for elastic_net
        """

        self.sklearn_params = kwargs
        self.alphas = alphas if alphas is not None else self.alphas
        self.num_subsamples = num_subsamples if num_subsamples is not None else self.num_subsamples

    def run_regression(self):

        MPControl.sync_processes("pre_stars")

        betas, resc_betas = StARS(self.design, self.response, self.random_seed, alphas=self.alphas,
                                  num_subsamples=self.num_subsamples,
                                  parameters=self.sklearn_params).run()

        MPControl.sync_processes("post_stars")

        return [betas], [resc_betas]
