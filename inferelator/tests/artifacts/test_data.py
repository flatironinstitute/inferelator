import pandas as pd
import scipy.sparse as sps
from inferelator.utils import InferelatorData


class TestDataSingleCellLike(object):
    expression_matrix = pd.DataFrame([[2, 28, 0, 16, 1, 3], [6, 21, 0, 3, 1, 3], [4, 39, 0, 17, 1, 3],
                                      [8, 34, 0, 7, 1, 3], [6, 26, 0, 3, 1, 3], [1, 31, 0, 1, 1, 4],
                                      [3, 27, 0, 5, 1, 4], [8, 34, 0, 9, 1, 3], [1, 22, 0, 3, 1, 4],
                                      [9, 33, 0, 17, 1, 2]],
                                     columns=["gene1", "gene2", "gene3", "gene4", "gene5", "gene6"]).transpose()
    meta_data = pd.DataFrame({"Condition": ["A", "B", "C", "C", "B", "B", "A", "C", "B", "C"],
                              "Genotype": ['WT', 'WT', 'WT', 'WT', 'WT', 'WT', 'WT', 'WT', 'WT', 'WT']})
    priors_data = pd.DataFrame([[0, 1], [0, 1], [1, 0], [0, 0]], index=["gene1", "gene2", "gene4", "gene5"],
                               columns=["gene3", "gene6"])
    gene_metadata = pd.DataFrame({"SystematicName": ["gene1", "gene2", "gene3", "gene4", "gene7", "gene6"]})
    gene_list_index = "SystematicName"
    tf_names = ["gene3", "gene6"]


TEST_DATA = InferelatorData(TestDataSingleCellLike.expression_matrix,
                            transpose_expression=True,
                            meta_data=TestDataSingleCellLike.meta_data,
                            gene_data=TestDataSingleCellLike.gene_metadata,
                            gene_data_idx_column=TestDataSingleCellLike.gene_list_index,
                            sample_names=list(map(str, range(10))))

TEST_DATA_SPARSE = InferelatorData(sps.csr_matrix(TestDataSingleCellLike.expression_matrix.T.values),
                                   gene_names=TestDataSingleCellLike.expression_matrix.index,
                                   sample_names=list(map(str, range(10))),
                                   meta_data=TestDataSingleCellLike.meta_data,
                                   gene_data=TestDataSingleCellLike.gene_metadata,
                                   gene_data_idx_column=TestDataSingleCellLike.gene_list_index)

CORRECT_GENES_INTERSECT = pd.Index(["gene1", "gene2", "gene3", "gene4", "gene6"])
CORRECT_GENES_NZ_VAR = pd.Index(["gene1", "gene2", "gene4", "gene6"])
