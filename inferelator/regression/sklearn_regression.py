import inspect
import itertools
import numpy as np
from sklearn.base import BaseEstimator

from inferelator.utils import (
    Debug,
    Validator as check,
    make_array_2d
)
from inferelator.distributed.inferelator_mp import MPControl
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


def sklearn_gene(
    x,
    y,
    model,
    min_coef=None,
    **kwargs
):
    """
    Use a scikit-learn model for regression

    :param x: Feature array
    :type x: np.ndarray [N x K]
    :param y: Response array
    :type y: np.ndarray [N x 1]
    :param model: Instance of a scikit BaseEstimator-derived model
    :type model: BaseEstimator
    :param min_coef: A minimum coefficient value to include in the model.
        Any values smaller will be set to 0.
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
        coefs[np.abs(coefs) < min_coef] = 0.

    # Create a boolean array where coefficients are nonzero [K, ]
    coef_nonzero = coefs != 0

    # If there are non-zero coefficients, redo the linear regression
    # with them alone and calculate beta_resc
    if coef_nonzero.sum() > 0:
        x = x[:, coef_nonzero]
        make_array_2d(y)
        betas = recalculate_betas_from_selected(x, y)
        betas_resc = predict_error_reduction(x, y, betas)
        return dict(
            pp=coef_nonzero,
            betas=betas,
            betas_resc=betas_resc
        )

    else:
        return dict(
            pp=np.repeat(True, K).tolist(),
            betas=np.zeros(K),
            betas_resc=np.zeros(K)
        )


class SKLearnRegression(BaseRegression):

    def __init__(self, x, y, model, random_state=None, **kwargs):
        self.params = kwargs

        if random_state is not None:
            self.params["random_state"] = random_state

        self.min_coef = self.params.pop("min_coef", None)
        self.model = model

        super(SKLearnRegression, self).__init__(x, y)

    def regress(self):
        """
        Execute Elastic Net

        :return: list
            Returns a list of regression results that base_regression's
            pileup_data can process
        """

        nG = self.G
        X = self.X.values

        return MPControl.map(
            _sklearn_regression_wrapper,
            itertools.repeat(X, nG),
            gene_data_generator(self.Y, nG),
            itertools.repeat(self.model, nG),
            itertools.repeat(self.genes, nG),
            itertools.repeat(nG, nG),
            range(self.G),
            params=self.params,
            min_coef=self.min_coef,
            scatter=[X]
        )


def _sklearn_regression_wrapper(
    X,
    y,
    model,
    genes,
    nG,
    j,
    params={},
    min_coef=None
):
        """ Wrapper for multiprocessing sklearn models """
        Debug.vprint(
            PROGRESS_STR.format(gn=genes[j], i=j, total=nG),
            level=0 if j % 1000 == 0 else 2
        )

        data = sklearn_gene(
            X,
            PreprocessData.preprocess_response_vector(y),
            model(**params),
            min_coef=min_coef
        )

        data['ind'] = j
        return data


class SKLearnWorkflowMixin(_RegressionWorkflowMixin):
    """
    Use any scikit-learn regression module
    """

    _sklearn_model = None
    _sklearn_model_params = None
    _sklearn_add_random_state = False

    def __init__(self, *args, **kwargs):
        self._sklearn_model_params = {}
        super(SKLearnWorkflowMixin, self).__init__(*args, **kwargs)

    def set_regression_parameters(
        self,
        model=None,
        add_random_state=None,
        **kwargs
    ):
        """
        Set parameters to use a sklearn model for regression

        :param model: A scikit-learn model class
        :type model: BaseEstimator subclass
        :param add_random_state: Flag to include workflow random seed
            as "random_state" in the model
        :type add_random_state: bool
        :param kwargs: Any arguments which should be passed to the
            scikit-learn model class instantiation
        :type kwargs: any
        """

        if model is not None and not inspect.isclass(model):
            raise ValueError(
                "Pass an uninstantiated scikit-learn model "
                "(i.e. LinearRegression, not LinearRegression()"
            )

        self._set_with_warning(
            "_sklearn_model",
            model
        )
        self._set_without_warning(
            "_sklearn_add_random_state",
            add_random_state
        )
        self._sklearn_model_params.update(
            kwargs
        )

    def run_bootstrap(self, bootstrap):
        x = self.design.get_bootstrap(bootstrap)
        y = self.response.get_bootstrap(bootstrap)

        Debug.vprint(
            'Calculating betas using SKLearn model '
            f'{self._sklearn_model.__name__}',
            level=0
        )

        if self._sklearn_add_random_state:
            seed = self.random_seed
        else:
            seed = None

        return SKLearnRegression(
            x,
            y,
            self._sklearn_model,
            random_state=seed,
            **self._sklearn_model_params
        ).run()


class SKLearnByTaskMixin(
    _MultitaskRegressionWorkflowMixin,
    SKLearnWorkflowMixin
):
    """
    This runs scikit models on tasks defined by the MTL workflow
    """

    def run_bootstrap(self, bootstrap_idx):
        betas, betas_resc = [], []

        if self._sklearn_add_random_state:
            seed = self.random_seed
        else:
            seed = None

        for k in range(self._n_tasks):

            # Select the appropriate bootstrap from each task
            # and stash the data into X and Y
            if bootstrap_idx is not None:
                x = self._task_design[k].get_bootstrap(
                    self._task_bootstraps[k][bootstrap_idx]
                )
                y = self._task_response[k].get_bootstrap(
                    self._task_bootstraps[k][bootstrap_idx]
                )
            else:
                x = self._task_design[k]
                y = self._task_response[k]

            Debug.vprint(
                f'Calculating task {k} using '
                f'{self._sklearn_model.__name__}',
                level=0
            )

            t_beta, t_br = SKLearnRegression(
                x,
                y,
                self._sklearn_model,
                random_state=seed,
                **self._sklearn_model_params
            ).run()

            betas.append(t_beta)
            betas_resc.append(t_br)

        return betas, betas_resc
