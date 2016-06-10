
import pandas as pd
from . import condition
from . import time_series

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
    
def separate_time_series(metadata_dicts, conditions_dict):
    "return a dictionary of time series and dictionary of non-timeseries conditions"
    time_series_dict = {}
    conditions_dict = conditions_dict.copy()  # copy to modify
    first_conditions = []
    for name in conditions_dict.keys():
        metadata = metadata_dicts[name]
        condition = conditions_dict[name]
        if metadata['is1stLast'] == "f":
            first_conditions.append(condition)
            #del conditions_dict[name]
    for first_condition in first_conditions:
        condition = first_condition
        name = condition.name
        ts = time_series.TimeSeries(first_condition)
        while name:
            del conditions_dict[name]
            metadata = metadata_dicts[name]
            prevname = metadata["prevCol"]
            if prevname:
                interval = metadata["del.t"]
                assert interval > 0, "time series interval must be positive."
                ts.add_condition(prevname, condition, interval)
            name = metadata["nextCol"]
            condition = conditions_dict.get(name)
        time_series_dict[first_condition.name] = ts
    # all remaining conditions should be "e" conditions_dict
    for name in conditions_dict:
        metadata = metadata_dicts[name]
        assert metadata["is1stLast"] == "e", "time series entry not classified " + repr(name)
    return (time_series_dict, conditions_dict)

def read_tf_names(file_like):
    "Read transcription factor names from one-column tsv file.  Return list of names."
    exp = pd.read_csv(file_like, sep="\t", header=None)
    assert exp.shape[1] == 1, "transcription factor file should have one column "
    return list(exp[0])