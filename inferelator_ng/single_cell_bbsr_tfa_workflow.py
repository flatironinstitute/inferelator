from inferelator_ng import bbsr_tfa_workflow, bbsr_python, utils, single_cell, tfa, mi
import gc
import sys
import time
import pandas as pd
import numpy as np

KVS_CLUSTER_KEY = 'cluster_idx'


class Single_Cell_BBSR_TFA_Workflow(bbsr_tfa_workflow.BBSR_TFA_Workflow):
    # Computed Data Structure
    cluster_index = None

    # Configuration parameters
    expression_matrix_compression = None

    def startup_controller(self):
        utils.kvs_async(self.kvs, chunk=self.async_chunk).execute_master_first(self.startup_run)

    def startup_run(self):
        self.get_data()
        self.compute_common_data()

    def startup_finish(self):
        utils.kvsTearDown(self.kvs, self.rank, kvs_key=KVS_CLUSTER_KEY)
        self.compute_activity()

    def compute_common_data(self):
        self.filter_expression_and_priors()

    def compute_activity(self):
        self.design = tfa.TFA(self.priors_data, self.expression_matrix,
                              self.expression_matrix).compute_transcription_factor_activity()
        self.response = self.expression_matrix

    """
    def compute_common_data(self):
        self.filter_expression_and_priors()

        # Run the clustering once and distribute it to avoid a nasty spike in memory usage
        if self.is_master():
            self.cluster_index = single_cell.initial_clustering(self.expression_matrix)
            self.kvs.put(KVS_CLUSTER_KEY, self.cluster_index)
        else:
            self.cluster_index = self.kvs.view(KVS_CLUSTER_KEY)
            
    


    def compute_activity(self):
        # Bulk up and normalize clusters
        bulk = single_cell.make_clusters_from_singles(self.expression_matrix, self.cluster_index)
        utils.Debug.vprint("Pseudobulk data matrix assembled [{}]".format(bulk.shape))

        # Calculate TFA and then break it back into single cells
        self.design = tfa.TFA(self.priors_data, bulk, bulk).compute_transcription_factor_activity()
        self.design = single_cell.make_singles_from_clusters(self.design, self.cluster_index,
                                                             columns=self.expression_matrix.columns)
        self.response = self.expression_matrix
    """

    def run_bootstrap(self, bootstrap):
        X = self.design.iloc[:, bootstrap]
        Y = self.response.iloc[:, bootstrap]

        #
        #boot_cluster_idx = self.cluster_index[bootstrap]
        #X_bulk = single_cell.make_clusters_from_singles(X, boot_cluster_idx)
        #Y_bulk = single_cell.make_clusters_from_singles(Y, boot_cluster_idx)

        #utils.Debug.vprint("Rebulked design {des} & response {res} data".format(des=X_bulk.shape, res=Y_bulk.shape))

        # Calculate CLR & MI if we're proc 0 or get CLR & MI from the KVS if we're not
        utils.Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=1)
        clr_mat, mi_mat = mi.MIDriver(kvs=self.kvs, rank=self.rank).run(X, Y)

        # Trying to get ahead of this memory fire
        X_bulk = Y_bulk = bootstrap = boot_cluster_idx = mi_mat = None
        gc.collect()

        utils.Debug.vprint('Calculating betas using BBSR', level=1)
        ownCheck = utils.ownCheck(self.kvs, self.rank, chunk=25)

        # Run the BBSR on this bootstrap
        betas, re_betas = bbsr_python.BBSR_runner().run(X, Y, clr_mat, self.priors_data, self.kvs, self.rank, ownCheck)

        # Trying to get ahead of this memory fire
        X = Y = clr_mat = None
        gc.collect()

        return betas, re_betas

    def read_expression(self):
        """
        Overload the workflow.workflowBase expression reader to force count data in as a uint with the smallest memory
        footprint possible. Also allows for importing and then concatenating multiple files.

        Sets self.expression_matrix with the data
        """

        if isinstance(self.expression_matrix_file, list):
            file_name_list = self.expression_matrix_file
        else:
            file_name_list = [self.expression_matrix_file]

        self.expression_matrix = None

        st = time.time()
        for file in file_name_list:
            if self.expression_matrix is None:
                self.expression_matrix = self._read_count_matrix_file(file)
            else:
                self._concat_count_matrix_df(self._read_count_matrix_file(file))
        et = int(time.time() - st)

        # Report on the result
        df_shape = self.expression_matrix.shape
        df_size = int(sys.getsizeof(self.expression_matrix) / 1000000)
        utils.Debug.vprint_all("Proc {r}: Single-cell data {s} read into memory ({m} MB in {t} sec)".format(r=self.rank,
                                                                                                            s=df_shape,
                                                                                                            m=df_size,
                                                                                                            t=et))

    def _read_count_matrix_file(self, file_name):

        tsv = dict(sep="\t", header=0, index_col=0, compression=self.expression_matrix_compression)

        file_name = self.input_path(file_name)
        data = pd.read_table(file_name, **tsv)
        data = data.apply(pd.to_numeric, downcast='unsigned')

        if self.expression_matrix_transpose:
            data = data.transpose()

        return data

    def _concat_count_matrix_df(self, data):

        if self.expression_matrix_transpose:
            assert self.expression_matrix.columns.equals(data.columns)
            newidx = self.expression_matrix.index.tolist().extend(data.index.tolist())
            self.expression_matrix = pd.DataFrame(np.concatenate((self.expression_matrix.values, data.values), axis=0),
                                                  index=newidx,
                                                  columns=data.columns)
        else:
            assert self.expression_matrix.index.equals(data.index)
            newidx = self.expression_matrix.columns.tolist().extend(data.columns.tolist())
            self.expression_matrix = pd.DataFrame(np.concatenate((self.expression_matrix.values, data.values), axis=1),
                                                  index=data.index,
                                                  columns=newidx)
