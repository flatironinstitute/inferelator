import numpy as np

from inferelator_ng import utils
from inferelator_ng import regression
from inferelator_ng.bayes_stats import predict_error_reduction
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
    (n, k) = X.shape
    X = X.T
    # Fit the linear model using the elastic net
    model = ElasticNetCV(**params).fit(X, Y.ravel())

    # Set coefficients below threshold to 0
    model.coef_[np.abs(model.coef_) < MIN_COEF] = 0.
    coef_nonzero = model.coef_ != 0

    if coef_nonzero.sum() > 0:
        idx = utils.bool_to_index(coef_nonzero)
        x = X[:, idx]
        best_betas = np.zeros(x.shape[1], dtype=np.dtype(float))
        beta_hat = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, Y))
        for i, j in enumerate(idx):
            best_betas[j] = beta_hat[i]
        betas_resc = predict_error_reduction(x, Y, best_betas)
        return dict(pp=idx,
                    betas=best_betas,
                    betas_resc=betas_resc)
    else:
        return dict(pp=np.repeat(True, k).tolist(),
                    betas=np.zeros(k),
                    betas_resc=np.zeros(k))


class ElasticNet(regression.BaseRegression):
    params = ELASTICNET_PARAMETERS

    def __init__(self, X, Y, kvs):
        # Get the IDs and total count for the genes and predictors
        self.K = X.shape[0]
        self.tfs = X.index.values.tolist()
        self.G = Y.shape[0]
        self.genes = Y.index.values.tolist()

        # Rescale input data
        self.X = self._scale(X)
        utils.Debug.vprint("Predictor matrix {} ready".format(X.shape))
        self.Y = self._scale(Y)
        utils.Debug.vprint("Response matrix {} ready".format(Y.shape))

    def run(self):
        """
        Execute BBSR separately on each response variable in the data

        :return: pd.DataFrame [G x K], pd.DataFrame [G x K]
            Returns the regression betas and beta error reductions for all threads if this is the master thread (rank 0)
            Returns None, None if it's a subordinate thread
        """
        regression_data = []

        # For every response variable G, check to see if this thread should run BBSR for that variable
        # If it should (ownCheck is TRUE), slice the data for that response variable
        # And send the values (as an ndarray because pd.df indexing is glacially slow) to bayes_stats.bbsr
        # Keep a list of the resulting regression results
        oc = self.kvs.own_check(chunk=self.chunk)
        for j in range(self.G):
            if next(oc):
                level = 0 if j % 100 == 0 else 2
                utils.Debug.vprint("Regression on {gn} [{i} / {total}]".format(gn=self.Y.index[j],
                                                                               i=j,
                                                                               total=self.G), level=level)
                data = elastic_net(self.X.values, self.Y.ix[j, :].values, self.params)
                data['ind'] = j
                regression_data.append(data)

        # Put the regression results that this thread has calculated into KVS
        self.kvs.put('plist', (self.kvs.rank, regression_data))

        # If this is the master thread, pile the regression betas into dataframes and return them
        if self.kvs.is_master:
            return self.pileup_data()
        else:
            return None, None


class ElasticNetRunner:
    def run(self, X, Y, kvs=None):
        return ElasticNet(X, Y, kvs).run()
