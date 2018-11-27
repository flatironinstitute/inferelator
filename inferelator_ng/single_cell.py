import numpy as np
import pandas as pd

def umi(data, axis=1):
    """
    :param data: pd.DataFrame [N x G]
    :param axis: int
    :return umi: pd.Series
    """
    return data.sum(axis=axis)
