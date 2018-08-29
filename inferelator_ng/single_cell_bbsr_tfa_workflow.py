from inferelator_ng import bbsr_tfa_workflow, bbsr_python, utils, single_cell, tfa, mi
import gc
import sys
import time
import pandas as pd
import numpy as np

KVS_CLUSTER_KEY = 'cluster_idx'

class Single_Cell_BBSR_TFA_Workflow(bbsr_tfa_workflow.BBSR_TFA_Workflow):
    cluster_index = None

    count_file_compression = None
    count_file_chunk_size = None

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        self.filter_expression_and_priors()

        # Run the clustering once and distribute it to avoid a nasty spike in memory usage
        if self.is_master():
            self.cluster_index = single_cell.initial_clustering(self.expression_matrix)
            self.kvs.put(KVS_CLUSTER_KEY, self.cluster_index)
        else:
            self.cluster_index = self.kvs.view(KVS_CLUSTER_KEY)
        utils.kvs_sync_processes(self.kvs, self.rank)
        utils.kvsTearDown(self.kvs, self.rank, kvs_key=KVS_CLUSTER_KEY)

    def compute_activity(self):
        # Bulk up and normalize clusters
        bulk = single_cell.make_clusters_from_singles(self.expression_matrix, self.cluster_index, pseudocount=True)
        utils.Debug.vprint("Pseudobulk data matrix assembled [{}]".format(bulk.shape))

        # Calculate TFA and then break it back into single cells
        self.design = tfa.TFA(self.priors_data, bulk, bulk).compute_transcription_factor_activity()
        self.design = single_cell.make_singles_from_clusters(self.design, self.cluster_index,
                                                             columns=self.expression_matrix.columns)
        self.response = self.expression_matrix

    def run_bootstrap(self, bootstrap):
        X = self.design.iloc[:, bootstrap]
        Y = self.response.iloc[:, bootstrap]
        boot_cluster_idx = self.cluster_index[bootstrap]

        X_bulk = single_cell.make_clusters_from_singles(X, boot_cluster_idx)
        Y_bulk = single_cell.make_clusters_from_singles(Y, boot_cluster_idx)

        utils.Debug.vprint("Rebulked design {des} & response {res} data".format(des=X_bulk.shape, res=Y_bulk.shape))

        # Calculate CLR & MI if we're proc 0 or get CLR & MI from the KVS if we're not
        utils.Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=1)
        clr_mat, mi_mat = mi.MIDriver(kvs=self.kvs, rank=self.rank).run(X_bulk, Y_bulk)

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
        footprint possible

        Sets self.expression_matrix.
        """

        # Set controller variables that will be needed to read stuff in
        csv = dict(sep="\t", header=0, index_col=0, compression=self.count_file_compression)
        dtype = np.dtype('uint16')
        file_name = self.input_path(self.expression_matrix_file)

        utils.Debug.vprint("Reading {f} file data".format(f=file_name))
        st = time.time()

        # If count_file_chunk_size is set, read it into memory chunkwise
        # This is significantly slower, but should cut peak memory usage a lot
        if self.count_file_chunk_size is not None:
            utils.Debug.vprint("Reading {f} file indexes".format(f=file_name))
            cols = pd.read_table(file_name, nrows=2, **csv).columns
            idx = pd.read_table(file_name, usecols=[0, 1], **csv).index

            self.expression_matrix = np.zeros((0, len(cols)), dtype=dtype)
            for i, chunk in enumerate(pd.read_table(file_name, chunksize=self.count_file_chunk_size, **csv)):
                self.expression_matrix = np.vstack((self.expression_matrix, chunk.values.astype(dtype)))
                utils.Debug.vprint("Processed row {i} of {l}".format(i=i*self.count_file_chunk_size, l=len(idx)), level=2)
            self.expression_matrix = pd.DataFrame(self.expression_matrix, index=idx, columns=cols)

        # If count_file_chunk_size isn't set, just read everything in with pandas and downcast afterwards
        else:
            self.expression_matrix = pd.read_table(file_name, **csv)
            self.expression_matrix = self.expression_matrix.apply(pd.to_numeric, downcast='unsigned')

        et = int(time.time() - st)

        # Report on the result
        df_shape = self.expression_matrix.shape
        df_size = int(sys.getsizeof(self.expression_matrix)/1000000)
        utils.Debug.vprint_all("Proc {r}: Single-cell data {s} read into memory ({m} MB in {t} sec)".format(r=self.rank,
                                                                                                            s=df_shape,
                                                                                                            m=df_size,
                                                                                                            t=et))
