from inferelator.regression import sklearn_regression
from sklearn.linear_model import ElasticNetCV
import copy

ELASTICNET_PARAMETERS = dict(l1_ratio=[0.5, 0.7, 0.9],
                             eps=0.001,
                             n_alphas=50,
                             alphas=None,
                             fit_intercept=True,
                             normalize=False,
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


class ElasticNetWorkflow(sklearn_regression.SKLearnWorkflow):
    """
    Add elasticnet regression into a workflow object
    """

    _sklearn_model = ElasticNetCV
    _sklearn_add_random_state = True

    def __init__(self, *args, **kwargs):
        super(ElasticNetWorkflow, self).__init__(*args, **kwargs)
        self._sklearn_model_params = copy.copy(ELASTICNET_PARAMETERS)


class ElasticNetByTaskRegressionWorkflow(sklearn_regression.SKLearnByTask):

    _sklearn_model = ElasticNetCV
    _sklearn_add_random_state = True

    def __init__(self, *args, **kwargs):
        super(ElasticNetByTaskRegressionWorkflow, self).__init__(*args, **kwargs)
        self._sklearn_model_params = copy.copy(ELASTICNET_PARAMETERS)
