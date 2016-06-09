

import pandas as pd

class Condition:

    """
    A condition maps gene names to numbers which often represent expression levels.
    
    Parameters
    ----------
    condition_name: str
        A unique name identifying the condition.
       
    gene_mapping: pd.Series
        A pandas Series holding the gene to number mapping or an object that
        can be converted to a pandas Series.
    """
    
    def __init__(self, condition_name, gene_mapping):
        self.name = condition_name
        self.gene_mapping = pd.Series(gene_mapping)