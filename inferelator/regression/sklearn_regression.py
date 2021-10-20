import numpy as np
from inferelator.utils import Validator as check
from inferelator import utils
from inferelator.regression import base_regression
from inferelator.distributed.inferelator_mp import MPControl
from sklearn.base import BaseEstimator
from inferelator.regression.base_regression import _MultitaskRegressionWorkflowMixin
import copy
import inspect


def sklearn_gene(x, y, model, min_coef=None, **kwargs):
    """
    Use a scikit-learn model for regression

    :param x: Feature array
    :type x: np.ndarray [N x K]
    :param y: Response array
    :type y: np.ndarray [N x 1]
    :param model: Instance of a scikit BaseEstimator-derived model
    :type model: BaseEstimator
    :param min_coef: A minimum coefficient value to include in the model. Any values smaller will be set to 0.
    :type min_coef: numeric
    :return: A dict of results for this gene
    :rtype: dict
    """
    assert check.argument_type(x, np.ndarray)
    assert check.argument_type(y, np.ndarray)
    assert check.argument_is_subclass(model, BaseEstimator)

    (N, K) = x.shape

    # Fit the model
    model.fit(x, y, **kwargs)

    # Get all model coefficients [K, ]
    try:
        coefs = model.coef_
    except AttributeError:
        coefs = model.estimator_.coef_

    # Set coefficients below threshold to 0
    if min_coef is not None:
        coefs[np.abs(coefs) < min_coef] = 0.  # Threshold coefficients

    coef_nonzero = coefs != 0  # Create a boolean array where coefficients are nonzero [K, ]

    # If there are non-zero coefficients, redo the linear regression with them alone
    # And calculate beta_resc
    if coef_nonzero.sum() > 0:
        x = x[:, coef_nonzero]
        utils.make_array_2d(y)
        betas = base_regression.recalculate_betas_from_selected(x, y)
        betas_resc = base_regression.predict_error_reduction(x, y, betas)
        return dict(pp=coef_nonzero,
                    betas=betas,
                    betas_resc=betas_resc)
    else:
        return dict(pp=np.repeat(True, K).tolist(),
                    betas=np.zeros(K),
                    betas_resc=np.zeros(K))


class SKLearnRegression(base_regression.BaseRegression):

    def __init__(self, x, y, prior_data, model, random_state=None, **kwargs):
        self.params = kwargs

        if random_state is not None:
            self.params["random_state"] = random_state

        self.min_coef = self.params.pop("min_coef", None)
        self.model = model(**self.params)

        super(SKLearnRegression, self).__init__(x, y, prior_data)

    def regress(self):
        """
        Execute Elastic Net

        :return: list
            Returns a list of regression results that base_regression's pileup_data can process
        """

        if MPControl.is_dask():
            return sklearn_regress_dask(self.X, self.Y, self.prior_mat, self.model, self.G, self.genes, self.min_coef)

        def regression_maker(j):
            level = 0 if j % 100 == 0 else 2
            utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=self.genes[j], i=j, total=self.G), level=level)

            X, Y = base_regression.PreprocessData.gene_preprocess(self.X.values,
                                                                  self.Y.get_gene_data(j, force_dense=True, flatten=True),
                                                                  self.prior_mat, self.genes[j])

            data = sklearn_gene(X, Y, copy.copy(self.model), min_coef=self.min_coef)
            data['ind'] = j
            return data

        return MPControl.map(regression_maker, range(self.G), tell_children=False)


class SKLearnWorkflowMixin(base_regression._RegressionWorkflowMixin):
    """
    Use any scikit-learn regression module
    """

    _sklearn_model = None
    _sklearn_model_params = None
    _sklearn_add_random_state = False

    def __init__(self, *args, **kwargs):
        self._sklearn_model_params = {}
        super(SKLearnWorkflowMixin, self).__init__(*args, **kwargs)

    def set_regression_parameters(self, model=None, add_random_state=None, **kwargs):
        """
        Set parameters to use a sklearn model for regression

        :param model: A scikit-learn model class
        :type model: BaseEstimator subclass
        :param add_random_state: Flag to include workflow random seed as "random_state" in the model
        :type add_random_state: bool
        :param kwargs: Any arguments which should be passed to the scikit-learn model class instantiation
        :type kwargs: any
        """

        if model is not None and not inspect.isclass(model):
            raise ValueError("Pass an uninstantiated scikit-learn model (i.e. LinearRegression, not LinearRegression()")

        self._set_with_warning("_sklearn_model", model)
        self._set_without_warning("_sklearn_add_random_state", add_random_state)
        self._sklearn_model_params.update(kwargs)

    def run_bootstrap(self, bootstrap):
        x = self.design.get_bootstrap(bootstrap)
        y = self.response.get_bootstrap(bootstrap)
        utils.Debug.vprint('Calculating betas using SKLearn model {m}'.format(m=self._sklearn_model.__name__), level=0)

        return SKLearnRegression(x, y, self.priors_data,
                                 self._sklearn_model,
                                 random_state=self.random_seed if self._sklearn_add_random_state else None,
                                 **self._sklearn_model_params).run()


class SKLearnByTaskMixin(_MultitaskRegressionWorkflowMixin, SKLearnWorkflowMixin):
    """
    This runs BBSR regression on tasks defined by the AMUSR regression (MTL) workflow
    """

    def run_bootstrap(self, bootstrap_idx):
        betas, betas_resc = [], []

        # Select the appropriate bootstrap from each task and stash the data into X and Y
        for k in range(self._n_tasks):
            x = self._task_design[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx])
            y = self._task_response[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx])

            # Make sure that the priors align to the expression matrix
            priors_data = self._task_priors[k].reindex(labels=self._targets, axis=0). \
                reindex(labels=self._regulators, axis=1). \
                fillna(value=0)

            utils.Debug.vprint('Calculating task {k} using {n}'.format(k=k, n=self._sklearn_model.__name__), level=0)
            t_beta, t_br = SKLearnRegression(x, y, priors_data,
                                             self._sklearn_model,
                                             random_state=self.random_seed if self._sklearn_add_random_state else None,
                                             **self._sklearn_model_params).run()
            betas.append(t_beta)
            betas_resc.append(t_br)

        return betas, betas_resc

def sklearn_regress_dask(X, Y, prior_data, model, G, genes, min_coef):
    """
    Execute regression (SKLearn)

    :return: list
        Returns a list of regression results that the pileup_data can process
    """
    assert MPControl.is_dask()

    from dask import distributed 
    from inferelator.distributed.dask_functions import DASK_SCATTER_TIMEOUT, process_futures_into_list

    DaskController = MPControl.client

    def regression_maker(j, x, y, priors):
        level = 0 if j % 100 == 0 else 2
        utils.Debug.allprint(base_regression.PROGRESS_STR.format(gn=genes[j], i=j, total=G), level=level)

        x, y = base_regression.PreprocessData.gene_preprocess(x, y, priors, genes[j])

        data = sklearn_gene(x, y, copy.copy(model), min_coef=min_coef)
        data['ind'] = j
        return j, data

    # Scatter common data to workers
    [scatter_x] = DaskController.client.scatter([X.values], broadcast=True, hash=False)
    [scatter_priors] = DaskController.client.scatter([prior_data], broadcast=True, hash=False)

    # Wait for scattering to finish before creating futures
    distributed.wait(scatter_x, timeout=DASK_SCATTER_TIMEOUT)
    distributed.wait(scatter_priors, timeout=DASK_SCATTER_TIMEOUT)

    future_list = [DaskController.client.submit(regression_maker, i, scatter_x,
                                                Y.get_gene_data(i, force_dense=True, flatten=True),
                                                scatter_priors)
                   for i in range(G)]

    # Collect results as they finish instead of waiting for all workers to be done
    result_list = process_futures_into_list(future_list)

    DaskController.client.cancel(scatter_x)
    DaskController.client.cancel(scatter_priors)

    return result_list