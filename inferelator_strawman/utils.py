
import pandas as pd
from . import condition

def df_from_tsv(file_like):
    "Read a tsv file or buffer with headers and row ids into a pandas dataframe."
    return pd.read_csv(file_like, sep="\t", header=0, index_col=0)
    
def conditions_from_df(data_frame):
    "Return a dictionary of named conditions from a pandas dataframe where the conditions are columns."
    result = {}
    for name in data_frame.columns:
        mapping = data_frame[name]
        cond = condition.Condition(name, mapping)
        result[name] = cond
    return result

def conditions_from_tsv(file_like):
    "Return a dictionary of named conditions from a TSV formatted file."
    data_frame = df_from_tsv(file_like)
    return conditions_from_df(data_frame)
    
def metadata_df(file_like):
    "Read a metadata file as a pandas data frame."
    return pd.read_csv(file_like, sep="\t", header=0, index_col="condName")
    
def metadata_dicts(data_frame):
    """Convert data frame to a dictionary mapping condition name to metadata dictionary.
    For time series entries add "nextCol" pointers.
    """
    dictionaries = {}
    data_frame_t = data_frame.transpose().fillna(False)
    for name in data_frame_t:
         d = data_frame_t[name].to_dict()
         d["nextCol"] = None
         dictionaries[name] = d
    # add "nextCol"
    for name in dictionaries:
        d = dictionaries[name]
        prev = d["prevCol"]
        if prev:
            prevd = dictionaries[prev]
            assert prevd.get("nextCol") is None, "previous condition overdefined"
            prevd["nextCol"] = name
    return dictionaries