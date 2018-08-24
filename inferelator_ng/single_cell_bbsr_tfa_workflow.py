from . import bbsr_workflow, bbsr_python, utils, single_cell, tfa, mi
import gc
import pandas as pd
import numpy as np


class Single_Cell_BBSR_TFA_Workflow(bbsr_workflow.BBSRWorkflow):

    cluster_index = None

    def __init__(self):
        # Read in the normal data files from BBSRWorkflow & Workflow
        super(Single_Cell_BBSR_TFA_Workflow, self).__init__()

    def preprocess_data(self):
        # Run the normal workflow preprocessing to read in data
        super(bbsr_workflow.BBSRWorkflow, self).preprocess_data()

        # Cluster single cells by pearson correlation distance
        self.cluster_index = single_cell.initial_clustering(self.expression_matrix)

        # Bulk up and normalize clusters
        bulk = single_cell.make_clusters_from_singles(self.expression_matrix, self.cluster_index, pseudocount=True)
        utils.Debug.vprint("Pseudobulk data matrix assembled [{}]".format(bulk.shape))

        # Calculate TFA and then break it back into single cells
        self.design = tfa.TFA(self.priors_data, bulk, bulk).compute_transcription_factor_activity()
        self.design = single_cell.make_singles_from_clusters(self.design,
                                                             self.cluster_index,
                                                             columns=self.expression_matrix.columns)
        self.response = self.expression_matrix

    def run_bootstrap(self, idx, bootstrap):
        utils.Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=1)

        X = self.design.iloc[:, bootstrap]
        Y = self.response.iloc[:, bootstrap]
        boot_cluster_idx = self.cluster_index[bootstrap]

        X_bulk = single_cell.make_clusters_from_singles(X, boot_cluster_idx)
        Y_bulk = single_cell.make_clusters_from_singles(Y, boot_cluster_idx)

        utils.Debug.vprint("Rebulked design {des} & response {res} data".format(des=X_bulk.shape, res=Y_bulk.shape))

        # Calculate CLR & MI if we're proc 0 or get CLR & MI from the KVS if we're not
        if self.process_mi_local:
            clr_mat, mi_mat = mi.MIDriver().run(X, Y)
        else:
            clr_mat, mi_mat = mi.MIDriver(kvs=self.kvs, rank=self.rank).run(X, Y)

        # Trying to get ahead of some memory leaks
        X_bulk, Y_bulk, bootstrap, boot_cluster_idx, mi_mat = None, None, None, None, None
        gc.collect()

        utils.Debug.vprint('Calculating betas using BBSR', level=1)
        ownCheck = utils.ownCheck(self.kvs, self.rank, chunk=25)

        # Run the BBSR on this bootstrap
        betas, re_betas = bbsr_python.BBSR_runner().run(X, Y, clr_mat, self.priors_data, self.kvs, self.rank, ownCheck)

        # Clear the MI data off the KVS
        if self.is_master():
            _ = self.kvs.get('mi %d' % idx)

        # Trying to get ahead of some memory leaks
        X, Y, idx, clr_mat = None, None, None, None
        gc.collect()

        return betas, re_betas

    def read_expression(self, dtype='uint16'):
        """
        Overload the workflow.workflowBase expression reader to force count data in as uint16
        """
        file_name = self.input_path(self.expression_matrix_file)

        cols = pd.read_csv(file_name, sep="\t", header=0, nrows=1, index_col=0).columns
        idx = pd.read_csv(file_name, sep="\t", header=0, usecols=[0, 1], index_col=0).index

        self.expression_matrix = pd.read_csv(file_name, sep="\t", header=None, usecols=range(len(cols) + 1)[1:],
                                             skiprows=1, index_col=None, dtype=dtype)
        self.expression_matrix.index = idx
        self.expression_matrix.columns = cols
