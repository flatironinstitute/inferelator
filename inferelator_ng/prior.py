import pybedtools
import pandas as pd

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
                 mode = 'closest', max_distance = 100000,
                 ignore_downstream = False, number_of_targets = 1):

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
        Possibility of dumping motifs associated with a particular edge (?)
        """

        # Reads and sort input BED file
        motifs = pybedtools.BedTool(self.motifs).sort()
        genes = pybedtools.BedTool(self.genes).sort()

        # Define edges in prior using method defined in self.mode
        edges = {}

        # For each motif, find closest gene within a certain window (self.max_distance) and that a prior interaction
        if self.mode == 'closest':
            # pybedtools wrapper around Bedtools closest function, D reports signed distance between motifs and genes
            assignments = motifs.closest(genes, D = 'b', k = self.number_of_targets, id = self.ignore_downstream) # id = True, for Yeast! what about k, is it something to optimize?
            # get index to retrieve important features later
            motif_start = 0
            target_idx = motifs.field_count()+3

        # For each target gene, finds motifs within a certain window (self.max_distance) and consider those as interactions in prior
        if self.mode == 'window':
            # pybedtools wrapper around Bedtools window function, w is the window around gene feature to be used
            assignments = genes.window(motifs, w = self.max_distance)
            # get index to retrieve important features later
            motif_start = genes.field_count()
            target_idx = 3
        
        motif_end = motif_start + motifs.field_count()-1
        
        # Loop over all assignments and define edges
        for assignment in assignments:
            # in the closest mode, one can only allow motifs that are within the distance set in max_distance
            if self.mode == 'closest':
                if abs(int(assignment[-1])) > self.max_distance:
                    continue
            # record edges as well as motifs associated with them
            assignment = assignment.fields
            motif = assignment[motif_start:motif_end]
            regulator = assignment[motif_start+3]
            target = assignment[target_idx]

            if regulator in edges:
                if target in edges[regulator]:
                    edges[regulator][target].append(motif)
                else:
                    edges[regulator][target] = [motif]
            else:
                edges[regulator] = {target : [motif]}
    
        # Make prior matrix
        prior = pd.DataFrame(0, index=self.targets, columns=self.regulators)
        # If there are multiple motifs for a TF assigned to the same gene, give larger weight to that interaction
        for regulator in edges:
            for target in edges[regulator]:
                if regulator in self.regulators and target in self.targets:
                    #weight = pybedtools.BedTool(edges[regulator][target]).merge().count()
                    weight = len(edges[regulator][target])
                    prior.ix[target, regulator] = weight
        return prior

