from . import workflow, design_response_translation, utils, mi, bbsr_python, results_processor

DEFAULT_delTmin = 0
DEFAULT_delTmax = 120
DEFAULT_tau = 60
DEFAULT_bs = 2


class BBSRWorkflow(workflow.WorkflowBase):
    delTmin = DEFAULT_delTmin
    delTmax = DEFAULT_delTmax
    tau = DEFAULT_tau
    num_bootstraps = DEFAULT_bs

    def __init__(self, delTmin=DEFAULT_delTmin, delTmax=DEFAULT_delTmax, tau=DEFAULT_tau, num_bootstraps=DEFAULT_bs):
        # Call out to super __init__
        super(BBSRWorkflow, self).__init__()

        self.delTmax = delTmax
        self.delTmin = delTmin
        self.tau = tau
        self.num_bootstraps = num_bootstraps

    def run(self):
        betas = []
        rescaled_betas = []

        # Bootstrap sample size is the number of experiments

        for idx, bootstrap in enumerate(self.get_bootstraps(self.design.shape[1], self.num_bootstraps)):
            utils.Debug.vprint('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps), level=0)
            current_betas, current_rescaled_betas = self.run_bootstrap(self.design.ix[:, bootstrap],
                                                                       self.response.ix[:, bootstrap],
                                                                       idx,
                                                                       bootstrap)
            if self.is_master():
                betas.append(current_betas)
                rescaled_betas.append(current_rescaled_betas)

        if self.is_master():
            self.emit_results(betas, rescaled_betas, self.gold_standard, self.priors_data)

    def run_bootstrap(self, X, Y, idx, bootstrap):
        """
        :param X: pd.DataFrame [m x b]
        :param Y: pd.DataFrame [m x b]
        :param idx: int
        :param bootstrap: list [n]
        :return betas, re_betas: pd.DataFrame [m x b], pd.DataFrame [m x b]
        """
        utils.Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=1)

        # Calculate CLR & MI if we're proc 0 or get CLR & MI from the KVS if we're not
        if self.is_master():
            clr_mat, _ = mi.MIDriver(cores=self.cores).run(X, Y)
            self.kvs.put('mi %d' % idx, clr_mat)
        else:
            clr_mat = self.kvs.view('mi %d' % idx)

        utils.Debug.vprint('Calculating betas using BBSR', level=1)
        ownCheck = utils.ownCheck(self.kvs, self.rank, chunk=25)

        # Run the BBSR on this bootstrap
        betas, re_betas = bbsr_python.BBSR_runner().run(X, Y, clr_mat, self.priors_data, self.kvs, self.rank, ownCheck)

        # Clear the MI data off the KVS
        if self.is_master():
            _ = self.kvs.get('mi %d' % idx)

        return betas, re_betas

    def preprocess_data(self):
        # Run preprocess data from WorkflowBase
        # Sets expression_matrix, tf_names, priors_data, meta_data, and gold_standard
        super(BBSRWorkflow, self).preprocess_data()

        # Set design, response, and half_tau_response
        self.compute_common_data()

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        utils.Debug.vprint('Creating design and response matrix ... ', level=0)
        drd = design_response_translation.PythonDRDriver()
        drd.delTmin, drd.delTmax, drd.tau = self.delTmin, self.delTmax, self.tau
        self.design, self.response = drd.run(self.expression_matrix, self.meta_data)

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        self.validate_output_path()
        rp = results_processor.ResultsProcessor(betas, rescaled_betas)
        rp.summarize_network(self.output_dir, gold_standard, priors)
