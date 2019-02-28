import numpy as np

from inferelator_ng import utils
from inferelator_ng import regression
from inferelator_ng.distributed.inferelator_mp import MPControl
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

    def run(self):

        """
        Execute Elastic Net

        :return: pd.DataFrame [G x K], pd.DataFrame [G x K]
            Returns the regression betas and beta error reductions for all threads if this is the master thread (rank 0)
            Returns None, None if it's a subordinate thread
        """

        def regression_maker(regression_obj, j):
            level = 0 if j % 100 == 0 else 2
            utils.Debug.vprint(regression.PROGRESS_STR.format(gn=self.genes[j], i=j, total=self.G), level=level)
            data = elastic_net(regression_obj.X.values, regression_obj.Y.iloc[j, :].values, regression_obj.params)
            data['ind'] = j
            return data

        dsk = {'j': list(range(self.G)), 'data': (regression_maker, self, 'j')}
        run_data = MPControl.get(dsk, 'data', tell_children=False)

        if MPControl.is_master:
            return self.pileup_data(run_data)
        else:
            return None, None


def patch_workflow(obj):
    """
    Add elasticnet regression into a TFAWorkflow object

    :param obj: TFAWorkflow
    """

    import types

    def run_bootstrap(self, bootstrap):
        X = self.design.iloc[:, bootstrap]
        Y = self.response.iloc[:, bootstrap]
        utils.Debug.vprint('Calculating betas using MEN', level=0)
        MPControl.sync_processes("pre-bootstrap")
        return ElasticNet(X, Y).run()

    obj.run_bootstrap = types.MethodType(run_bootstrap, obj)
