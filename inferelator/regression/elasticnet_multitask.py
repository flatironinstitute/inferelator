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

        # Select the appropriate bootstrap from each task and stash the data into X and Y
        for k in range(self._n_tasks):
            X = self._task_design[k].iloc[:, self._task_bootstraps[k][bootstrap_idx]].loc[self._regulators, :]
            Y = self._task_response[k].iloc[:, self._task_bootstraps[k][bootstrap_idx]].loc[self._targets, :]

            MPControl.sync_processes(pref="en_pre")

            utils.Debug.vprint('Calculating task {k} betas using MEN'.format(k=k), level=0)
            t_beta, t_br = ElasticNet(X, Y, random_seed=self.random_seed).run()
            betas.append(t_beta)
            betas_resc.append(t_br)

        return betas, betas_resc
