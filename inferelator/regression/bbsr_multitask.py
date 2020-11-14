import pandas as pd

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.utils import Debug
from inferelator.regression.base_regression import _MultitaskRegressionWorkflowMixin
from inferelator.regression.bbsr_python import BBSR, BBSRRegressionWorkflowMixin


class BBSRByTaskRegressionWorkflowMixin(_MultitaskRegressionWorkflowMixin, BBSRRegressionWorkflowMixin):
    """
    This runs BBSR regression on tasks defined by the AMUSR regression (MTL) workflow
    """

    def run_bootstrap(self, bootstrap_idx):
        betas, betas_resc = [], []

        # Select the appropriate bootstrap from each task and stash the data into X and Y
        for k in range(self._n_tasks):
            X = self._task_design[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx])
            Y = self._task_response[k].get_bootstrap(self._task_bootstraps[k][bootstrap_idx])

            # Make sure that the priors align to the expression matrix
            priors_data = self._task_priors[k].reindex(labels=self._targets, axis=0). \
                reindex(labels=self._regulators, axis=1). \
                fillna(value=0)

            if self.clr_only:
                # Create a mock prior with no information if clr_only is set
                priors_data = pd.DataFrame(0, index=priors_data.index, columns=priors_data.columns)

            MPControl.sync_processes(pref="bbsr_pre")

            Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=0)
            clr_matrix, _ = self.mi_driver().run(Y, X, return_mi=False)

            Debug.vprint('Calculating task {k} betas using BBSR'.format(k=k), level=0)
            t_beta, t_br = BBSR(X, Y, clr_matrix, priors_data,
                                prior_weight=self.prior_weight, no_prior_weight=self.no_prior_weight,
                                nS=self.bsr_feature_num).run()
            betas.append(t_beta)
            betas_resc.append(t_br)

        return betas, betas_resc
