from inferelator.distributed.inferelator_mp import MPControl
from inferelator import utils

from inferelator.regression.amusr_regression import AMUSRRegressionWorkflow
from inferelator.regression.elasticnet_python import ElasticNet, ElasticNetWorkflow


class ElasticNetByTaskRegressionWorkflow(AMUSRRegressionWorkflow, ElasticNetWorkflow):
    """
    This runs BBSR regression on tasks defined by the AMUSR regression (MTL) workflow
    """

    def run_bootstrap(self, bootstrap_idx):
        betas, betas_resc = [], []

        # Make sure that the priors align to the expression matrix
        priors_data = self.priors_data.reindex(labels=self.targets, axis=0).fillna(value=0)
        priors_data = priors_data.reindex(labels=self.regulators, axis=1).fillna(value=0)

        # Select the appropriate bootstrap from each task and stash the data into X and Y
        for k in range(self.n_tasks):
            X = self.task_design[k].iloc[:, self.task_bootstraps[k][bootstrap_idx]].loc[self.regulators, :]
            Y = self.task_response[k].iloc[:, self.task_bootstraps[k][bootstrap_idx]].loc[self.targets, :]

            MPControl.sync_processes(pref="en_pre")

            utils.Debug.vprint('Calculating task {k} betas using BBSR'.format(k=k), level=0)
            t_beta, t_br = ElasticNet(X, Y, random_seed=self.random_seed).run()
            betas.append(t_beta)
            betas_resc.append(t_br)

        return betas, betas_resc
