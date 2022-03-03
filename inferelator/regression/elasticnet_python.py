from inferelator.regression import sklearn_regression
from sklearn.linear_model import ElasticNetCV
import copy

ELASTICNET_PARAMETERS = dict(l1_ratio=[0.5, 0.7, 0.9],
                             eps=0.001,
                             n_alphas=50,
                             alphas=None,
                             fit_intercept=True,
                             precompute='auto',
                             max_iter=1000,
                             tol=0.001,
                             cv=3,
                             copy_X=True,
                             verbose=0,
                             n_jobs=1,
                             positive=False,
                             selection='random',
                             min_coef=0.1)


class ElasticNetWorkflowMixin(sklearn_regression.SKLearnWorkflowMixin):
    """
    Set default parameters to run scikit-learn ElasticNetCV
    """

    _sklearn_model = ElasticNetCV
    _sklearn_add_random_state = True

    def __init__(self, *args, **kwargs):
        super(ElasticNetWorkflowMixin, self).__init__(*args, **kwargs)
        self._sklearn_model_params = copy.copy(ELASTICNET_PARAMETERS)


class ElasticNetByTaskRegressionWorkflowMixin(sklearn_regression.SKLearnByTaskMixin):

    _sklearn_model = ElasticNetCV
    _sklearn_add_random_state = True

    def __init__(self, *args, **kwargs):
        super(ElasticNetByTaskRegressionWorkflowMixin, self).__init__(*args, **kwargs)
        self._sklearn_model_params = copy.copy(ELASTICNET_PARAMETERS)
