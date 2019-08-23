import pandas as pd


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
