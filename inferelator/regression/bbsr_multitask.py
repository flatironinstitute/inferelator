from inferelator.distributed.inferelator_mp import MPControl
from inferelator import utils

from inferelator.regression.amusr_regression import AMUSRRegressionWorkflow
from inferelator.regression.bbsr_python import BBSR, BBSRRegressionWorkflow


class BBSRByTaskRegressionWorkflow(AMUSRRegressionWorkflow, BBSRRegressionWorkflow):
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

            MPControl.sync_processes(pref="bbsr_pre")

            utils.Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=0)
            clr_matrix, mi_matrix = self.mi_driver(sync_in_tmp_path=self.mi_sync_path).run(X, Y)
            mi_matrix = None

            utils.Debug.vprint('Calculating task {k} betas using BBSR'.format(k=k), level=0)
            t_beta, t_br = BBSR(X, Y, clr_matrix, priors_data, prior_weight=self.prior_weight,
                                no_prior_weight=self.no_prior_weight, nS=self.bsr_feature_num).run()
            betas.append(t_beta)
            betas_resc.append(t_br)

        return betas, betas_resc
