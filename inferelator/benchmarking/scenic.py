from inferelator import utils
from inferelator.workflows.single_cell_workflow import SingleCellWorkflow
from inferelator.regression.base_regression import _RegressionWorkflowMixin
from inferelator.distributed.inferelator_mp import MPControl

import numpy as np
import pandas as pd

import tempfile
import os

# These are required to run this module but nothing else
# They are therefore not package dependencies
from arboreto.algo import grnboost2, genie3

from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df
from pyarrow.feather import write_feather

import scanpy as sc

ADJ_METHODS = {"grnboost2": grnboost2, "genie3": genie3}

FEATHER_FILE_NAME = "RANKED.gene_based.max.feather"
MOTIF_TABLE_NAME = "motifs-from-binary-prior.tbl"

MOTIF_TABLE_COLS = ['#motif_id',
 'motif_name',
 'motif_description',
 'source_name',
 'source_version',
 'gene_name',
 'motif_similarity_qvalue',
 'similar_motif_id',
 'similar_motif_description',
 'orthologous_identity',
 'orthologous_gene_name',
 'orthologous_species',
 'description']

MOTIF_TABLE_DEFAULTS = {'source_name': "binary_prior",
'source_version': 1.0,
'motif_similarity_qvalue': 0.0, 
'similar_motif_id': None,
'similar_motif_description': None,
'orthologous_identity': 1.0,
'orthologous_gene_name': None,
'orthologous_species': None,
'description': "Gene"}

MOTIF_NAME_COLS = ['#motif_id',
 'motif_name',
 'motif_description',
 'source_name',
 'source_version',
 'gene_name']

class SCENICWorkflow(SingleCellWorkflow):
    
    dask_temp_path = None
    _tmp_handle = None

    _feather_rank_file = None
    _motif_link_table_file = None

    @property
    def tmp_dir(self):

        if self._tmp_handle is None:
            self._tmp_handle = tempfile.TemporaryDirectory(prefix="SCENIC_", dir=self.dask_temp_path)
        
        return self._tmp_handle.name

    def startup_finish(self):

        self.align_priors_and_expression()

        tf_names = self.tf_names if self.tf_names is not None else self.priors_data.columns
        self.tf_names = [t for t in tf_names if t in self.data.gene_names]

        utils.Debug.vprint("Generating SCENIC prior files", level=0)

        self._feather_rank_file = self.create_feather_file_from_prior()
        self._motif_link_table_file = self.create_motif_table_from_prior()

        utils.Debug.vprint("Preprocessing data")

        sc.pp.filter_cells(self.data._adata, min_genes=200)
        sc.pp.filter_genes(self.data._adata, min_cells=3)

        self.data.convert_to_float()

        sc.pp.normalize_per_cell(self.data._adata, counts_per_cell_after=1e4)
        sc.pp.log1p(self.data._adata)
        sc.pp.scale(self.data._adata, max_value=10)


    def create_feather_file_from_prior(self):

        # Get rid of TFs which have no edges
        new_prior = self.priors_data.loc[:, (self.priors_data != 0).sum(axis=0) > 0]

        # Make sure to include all genes
        new_prior = new_prior.reindex(self.data.gene_names, axis=0).fillna(0).T.astype(int)
        new_prior.index.name = 'features'

        for i in range(new_prior.shape[0]):
            new_prior.iloc[i, :] = scenic_ranking_prior(new_prior.iloc[i, :], seed=42 + i).astype(int)

        new_prior.reset_index(inplace=True)
        feather_file = os.path.join(self.tmp_dir, FEATHER_FILE_NAME)
        write_feather(new_prior, feather_file)

        return feather_file

    def create_motif_table_from_prior(self):

        motif_table_file = os.path.join(self.tmp_dir, MOTIF_TABLE_NAME)

        mt = pd.DataFrame(columns=MOTIF_TABLE_COLS)

        for col in MOTIF_NAME_COLS:
            mt[col] = self.tf_names

        for col, val in MOTIF_TABLE_DEFAULTS.items():
            mt[col] = val

        mt.to_csv(motif_table_file, sep="\t", index=False)

        return motif_table_file
            

class SCENICRegression(_RegressionWorkflowMixin):

    adjacency_method = "grnboost2"
    do_scenic = True

    def run_regression(self):
        
        data_df = self.data.to_df()

        utils.Debug.vprint("Calculating {m} adjacencies".format(m=self.adjacency_method), level=0)

        # Get adjacencies
        adj_method = ADJ_METHODS[self.adjacency_method]

        if MPControl.is_dask():
            client_or_address = MPControl.client.client
            MPControl.client.check_cluster_state()
        else:
            client_or_address = 'local'

        adjacencies = adj_method(data_df, tf_names=self.tf_names, verbose=True, client_or_address=client_or_address,
                                 seed=self.random_seed)

        if self.do_scenic:

            # Convert adjacencies to modules
            modules = list(modules_from_adjacencies(adjacencies, data_df))

            # Load feather (rank) databases
            dbs = [RankingDatabase(fname = self._feather_rank_file, name = "RANKING_PRIOR")]

            utils.Debug.vprint("Pruning adjacencies with SCENIC", level=0)

            # Prune to df
            df = prune2df(dbs, modules, self._motif_link_table_file, client_or_address=client_or_address)

            return self.reprocess_scenic_output_to_inferelator_results(df, self.priors_data)

        else:

            return self.reprocess_adj_to_inferelator_results(adjacencies)

            
    @staticmethod
    def reprocess_scenic_output_to_inferelator_results(scenic_df, prior_data):

        # if there's nothing in the scenic output make an empty dataframe of 0s
        if scenic_df.shape[0] == 0:
            mat = pd.DataFrame(0.0, index=prior_data.index, columns=prior_data.columns)

        else:
            scenic_df = scenic_df.copy()
            scenic_df.index = scenic_df.index.droplevel(1)
            scenic_df.columns = scenic_df.columns.droplevel(0)

            mat = [pd.DataFrame(data).set_index(0).rename({1: tf}, axis=1)
                    for tf, data in scenic_df['TargetGenes'].iteritems()]

            mat = pd.concat(mat, axis=0).fillna(0)
            mat = mat.groupby(mat.index).agg('max')
            mat = mat.reindex(prior_data.columns, axis=1).reindex(prior_data.index, axis=0).fillna(0)
    
        return [mat], [mat.copy()], mat.copy(), mat.copy()

    @staticmethod
    def reprocess_adj_to_inferelator_results(adj):
        mat = adj.pivot(index='target', columns='TF', values='importance').fillna(0.)

        return [mat], [mat.copy()], mat.copy(), mat.copy()

# This code is lifted from https://github.com/aertslab/create_cisTarget_databases/cistarget_db.py
# It is not reimplemented in order to ensure that the methodology for ranking is identical
# There's no license on this code so this is just straight up theft
# Maybe put licenses on your academic code?
# You can have it back if you want. I really don't want it.

def scenic_ranking_prior(scores_series, seed):
    rng = np.random.default_rng(seed=seed)
    scores_numpy_array = scores_series.values
    random_permutations_to_break_ties_numpy = rng.permutation(scores_numpy_array.shape[0])
    ranking_with_broken_ties_for_motif_or_track_numpy = random_permutations_to_break_ties_numpy[
        (-scores_numpy_array)[random_permutations_to_break_ties_numpy].argsort()
    ].argsort().astype(scores_numpy_array.dtype)
    return pd.Series(ranking_with_broken_ties_for_motif_or_track_numpy, index=scores_series.index)
