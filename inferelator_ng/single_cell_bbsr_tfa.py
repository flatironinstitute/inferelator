import numpy as np
import datetime
import os

from kvsstcp.kvsclient import KVSClient
from . import workflow, utils, mi_clr_python, bbsr_python, single_cell, tfa, results_processor

DEFAULT_delTmin = 0
DEFAULT_delTmax = 120
DEFAULT_tau = 45

# Dial into KVS
kvs = KVSClient()
# Find out which process we are (assumes running under SLURM).
rank = int(os.environ['SLURM_PROCID'])

class Single_Cell_BBSR_TFA_Workflow(workflow.WorkflowBase):

    delTmin = DEFAULT_delTmin
    delTmax = DEFAULT_delTmax
    tau = DEFAULT_tau

    def __init__(self):

        # Call out to WorkflowBase __init__
        super(Single_Cell_BBSR_TFA_Workflow, self).__init__()

        # Set the random seed in np.random
        np.random.seed(self.random_seed)

        # Instantiate the driver classes
        self.mi = mi_clr_python.MIDriver()
        self.bbsr = bbsr_python.BBSR_runner()


    def run(self):

        # Get the data for regression from files
        self.get_data()

        # Creates each of the following data structures:
        # expression_matrix: pd.DataFrame (from pd.read_csv) - rows are genes, columns are experiments
        # tf_names: list (from list(pd.read_csv))
        # meta_data: pd.DataFrame (from pd.read_csv) - row for each experiment
        # priors_data: pd.DataFrame - row for each gene, column for each TF
        # gold_standard: pd.DataFrame - row for each gene, column for each TF

        # Make sure that the expression and priors data can be lined up
        self.filter_expression_and_priors()

        # Cluster and bulk up single cells to cluster
        self.bulk, self.cluster_index = single_cell.initial_clustering(self.expression_matrix)
        utils.Debug.vprint("Pseudobulk data matrix assembled [{}]".format(self.bulk.shape))

        # Calculate TFA and then break it back into single cells
        self.bulk = self.bulk.apply(single_cell._library_size_normalizer, axis=0, raw=True)
        self.activity = tfa.TFA(self.priors_data, self.bulk, self.bulk).compute_transcription_factor_activity()
        self.activity = single_cell.declustering(self.activity, self.cluster_index,
                                                 columns=self.expression_matrix.columns)

        self.response = self.expression_matrix

        betas = []
        rescaled_betas = []

        # Bootstrap sample size is the number of experiments

        for idx, bootstrap in enumerate(self.get_bootstraps()):

            print('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps))

            # X and Y are resampled based on bootstrap (index generated from np.random.choice(num_cols))
            X = self.activity.ix[:, bootstrap]
            Y = self.response.ix[:, bootstrap]

            boot_cluster_idx = self.cluster_index[bootstrap]
            X_bulk = single_cell.reclustering(X, boot_cluster_idx).apply(single_cell._library_size_normalizer,
                                                                         axis=0, raw=True)
            Y_bulk = single_cell.reclustering(Y, boot_cluster_idx).apply(single_cell._library_size_normalizer,
                                                                         axis=0, raw=True)

            print('Calculating MI, Background MI, and CLR Matrix')

            # Calculate CLR & MI if we're proc 0 or get CLR & MI from the KVS if we're not
            if 0 == rank:
                (clr_matrix, mi_matrix) = self.mi.run(X_bulk, Y_bulk)
                kvs.put('mi %d' % idx, (clr_matrix, mi_matrix))
            else:
                (clr_matrix, mi_matrix) = kvs.view('mi %d' % idx)

            print('Calculating betas using BBSR')

            # Create the generator to handle interprocess communication through KVS
            ownCheck = utils.ownCheck(kvs, rank, chunk=25)

            # Run the BBSR on this bootstrap
            X = single_cell.ss_df_norm(X)
            Y = single_cell.ss_df_norm(Y)
            current_betas, current_rescaled_betas = self.bbsr.run(X, Y, clr_matrix, self.priors_data, kvs,
                                                                               rank, ownCheck)

            if rank: continue
            betas.append(current_betas)
            rescaled_betas.append(current_rescaled_betas)

        self.emit_results(betas, rescaled_betas, self.gold_standard, self.priors_data)

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        if 0 == rank:
            output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(output_dir)
            self.results_processor = results_processor.ResultsProcessor(betas, rescaled_betas)
            self.results_processor.summarize_network(output_dir, gold_standard, priors)