import pybedtools
import pandas as pd
import numpy as np


class Prior:
    """
    Prior generates a prior matrix from motifs and gene coordinates
    Parameters:
    ----------------
    motifs_file: path/to/motifs BED format
    genes_file: path/to/genes BED format (TSS or gene body)
    targets: list of target genes
    regulators: list of regulators
    mode: closest (find closest target gene for each motif)
          window (finds motifs within a window of target gene feature)
    max_distance: maximum distance allowed for an assignment motif --> gene to be valid
    ignore_downstream: valid for closest mode only;
                       motifs can only be assigned to target gene if upstream (true in yeast, for example)
    number_of_targets: valid for closest mode only;
                       allow a motif to regulate 'number_of_targets' closest genes
    """

    def __init__(self, motifs_file, genes_file, targets, regulators,
                 mode='closest', max_distance=100000,
                 ignore_downstream=False, number_of_targets=1):

        self.genes = genes_file
        self.motifs = motifs_file
        self.targets = targets
        self.regulators = regulators
        self.mode = mode
        self.max_distance = max_distance
        self.number_of_targets = number_of_targets
        self.ignore_downstream = ignore_downstream

    def make_prior(self):

        """
        Returns a prior matrix, where non-zero entries are potential regulatory interactions between
        TFs (columns) and target genes (rows) - weights are set as the number of motifs associated with interaction
        """
        # Reads and sort input BED file
        genes = pybedtools.BedTool(self.genes).sort()
        motifs = pybedtools.BedTool(self.motifs).sort()

        # this step can take a long time...
        #        motifs_filt = []

        #        for idx, group in motifs.groupby('name'):
        #            group = pybedtools.BedTool.from_dataframe(group)
        #            group = group.merge(o='distinct', c=4).to_dataframe()
        #            motifs_filt.append(group)

        #        motifs = pybedtools.BedTool.from_dataframe(pd.concat(motifs_filt)).sort()

        # For each motif, find closest gene within a certain window (self.max_distance) and that a prior interaction
        if self.mode == 'closest':
            # pybedtools wrapper around Bedtools closest function, D reports signed distance between motifs and genes
            assignments = motifs.closest(genes, D='b', k=self.number_of_targets,
                                         id=self.ignore_downstream).to_dataframe()  # id = True, for Yeast! what about k, is it something to optimize?
            assignments = assignments.loc[assignments.iloc[:, -1].abs() <= self.max_distance, :]
            # get index to retrieve important features later
            motif_start = 0
            target_idx = motifs.field_count() + 3

        # For each target gene, finds motifs within a certain window (self.max_distance) and consider those as interactions in prior
        if self.mode == 'window':
            # pybedtools wrapper around Bedtools window function, w is the window around gene feature to be used
            assignments = genes.window(motifs, w=self.max_distance).to_dataframe()
            # get index to retrieve important features later
            motif_start = genes.field_count()
            target_idx = 3

        # Find column index for the regulator and target ids
        regulator = assignments.columns[motif_start + 3]
        target = assignments.columns[target_idx]

        # Count number of motifs associated with each target gene
        assignments = assignments.groupby([regulator, target]).size().reset_index()
        assignments.columns = ['regulator', 'target', 'interaction']
        assignments = assignments.loc[assignments.regulator.isin(self.regulators), :]
        assignments = assignments.loc[assignments.target.isin(self.targets), :]

        # Make prior matrix
        prior = pd.pivot_table(assignments, index='target', columns='regulator', values='interaction', fill_value=0)
        prior = prior.loc[:, self.regulators]
        prior = prior.loc[self.targets, :]
        if len(prior.columns) > 0:
            del prior.columns.name
        if len(prior.index) > 0:
            del prior.index.name
        prior.replace(np.nan, 0, inplace=True)
        return prior.astype(int)