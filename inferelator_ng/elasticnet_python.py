import numpy as np

from inferelator_ng import utils
from inferelator_ng import regression
from inferelator_ng import tfa_workflow
from sklearn.linear_model import ElasticNetCV

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
                             random_state=99,
                             selection='random')

MIN_COEF = 0.1


def elastic_net(X, Y, params):
    """

    :param X: np.ndarray [K x N]
    :param Y: np.ndarray [1 x N]
    :param params: dict
    :return:
    """
    (K, N) = X.shape
    X = X.T  # Make X into [N, K]
    Y = Y.flatten()  # Make Y into [N, ]

    # Fit the linear model using the elastic net
    model = ElasticNetCV(**params).fit(X, Y)

    # Set coefficients below threshold to 0
    coefs = model.coef_  # Get all model coefficients [K, ]
    coefs[np.abs(coefs) < MIN_COEF] = 0.  # Threshold coefficients
    coef_nonzero = coefs != 0  # Create a boolean array where coefficients are nonzero [K, ]

    # If there are non-zero coefficients, redo the linear regression with them alone
    # And calculate beta_resc
    if coef_nonzero.sum() > 0:
        x = X[:, coef_nonzero]
        utils.make_array_2d(Y)
        betas = regression.recalculate_betas_from_selected(x, Y)
        betas_resc = regression.predict_error_reduction(x, Y, betas)
        return dict(pp=coef_nonzero,
                    betas=betas,
                    betas_resc=betas_resc)
    else:
        return dict(pp=np.repeat(True, K).tolist(),
                    betas=np.zeros(K),
                    betas_resc=np.zeros(K))


class ElasticNet(regression.BaseRegression):
    params = ELASTICNET_PARAMETERS

    def regress(self, idx):
        return elastic_net(self.X.values, self.Y.ix[idx, :].values, self.params)


class ElasticNetRunner:
    def run(self, X, Y, kvs):
        return ElasticNet(X, Y, kvs).run()


class MEN_Workflow(tfa_workflow.TFAWorkFlow):
    # Drivers
    regression_driver = ElasticNetRunner

    def run_bootstrap(self, bootstrap):
        X = self.design.iloc[:, bootstrap]
        Y = self.response.iloc[:, bootstrap]
        utils.Debug.vprint('Calculating betas using MEN', level=0)
        self.kvs.sync_processes("pre-bootstrap")
        return self.regression_driver().run(X, Y, self.kvs)
