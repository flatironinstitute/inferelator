from . import bbsr_workflow, utils
from inferelator_ng.tfa import TFA


# This is a WorkflowBase with BBSR & TFA specific addons
class BBSR_TFA_Workflow(bbsr_workflow.BBSRWorkflow):

    activity = None

    def __init__(self):
        super(BBSR_TFA_Workflow, self).__init__()

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        super(BBSR_TFA_Workflow, self).run()

    def preprocess_data(self):
        # Run preprocess data from WorkflowBase and BBSRWorkflow
        super(BBSR_TFA_Workflow, self).preprocess_data()
        utils.Debug.vprint('Computing Transcription Factor Activity ... ', level=0)
        self.design = TFA(self.priors_data, self.design, self.half_tau_response).compute_transcription_factor_activity()
