import functools
import pandas as pd
import numpy as np
from scipy import sparse
import pandas.api.types as pat

from inferelator.utils import Debug


# Numpy / scipy matrix math function
# that's sparse-safe
def dot_product(
    a,
    b,
    dense=True,
    cast=True
):
    """
    Dot product two matrices together.
    Allow either matrix (or both or neither) to be sparse.

    :param a: Array A
    :param b: Array B
    :param dense: Always return a dense array
    :param cast: Unused
    :return: A @ B array
    :rtype: np.ndarray, sp.sparse.csr_matrix
    """
    if sparse.isspmatrix(a) and sparse.isspmatrix(b):
        return a.dot(b).A if dense else a.dot(b)
    elif sparse.isspmatrix(a) and dense:
        _arr = a.dot(b)
        return _arr.A if sparse.isspmatrix(_arr) else _arr
    elif sparse.isspmatrix(a) or sparse.isspmatrix(b):
        return a @ b
    else:
        return np.dot(a, b)


class DotProduct:

    _dot_func = dot_product

    @classmethod
    def set_mkl(cls, mkl=True):

        # If the MKL flag is None, don't change anything
        if mkl is None:
            pass

        # If the MKL flag is True, use the dot_product_mkl function
        # when .dot() is called
        if mkl:
            try:
                from sparse_dot_mkl import (
                    get_version_string,
                    dot_product_mkl
                )

                vstring = get_version_string()

                if vstring is None:
                    vstring = "Install mkl-service for details"

                Debug.vprint(
                    "Matrix multiplication will use sparse_dot_mkl "
                    "package with MKL: {vstring}",
                    level=2
                )

                cls._dot_func = dot_product_mkl

            # If it isn't available, use the
            # scipy/numpy functions instead
            except ImportError as err:
                Debug.vprint(
                    "Unable to load MKL with sparse_dot_mkl: " +
                    str(err),
                    level=0
                )

                cls._dot_func = dot_product

        # If the MKL flag is False, use the python (numpy/scipy)
        # functions when .dot() is called
        else:
            Debug.vprint(
                "Matrix multiplication will use numpy; "
                "this is not advised for sparse data",
                level=2
            )
            cls._dot_func = dot_product

    @classmethod
    def dot(cls, *args, **kwargs):
        return cls._dot_func(*args, **kwargs)


def df_from_tsv(
        file_like,
        has_index=True
):
    """
    Read a tsv file or buffer with headers
    and row ids into a pandas dataframe.
    """

    return pd.read_csv(
        file_like,
        sep="\t",
        header=0,
        index_col=0 if has_index else False
    )


def df_set_diag(
        df,
        val,
        copy=True
):
    """
    Sets the diagonal of a dataframe to a value.
    Diagonal in this case is anything where row label == column label.

    :param df: pd.DataFrame
        DataFrame to modify
    :param val: numeric
        Value to insert into any cells where row label == column label
    :param copy: bool
        Force-copy the dataframe instead of modifying in place
    :return: pd.DataFrame / int
        Return either the modified dataframe (if copied) or
        the number of cells modified (if changed in-place)
    """

    # Find all the labels that are shared between rows and columns
    isect = df.index.intersection(df.columns)

    if copy:
        df = df.copy()

    # Set the value where row and column names are the same
    for i in range(len(isect)):
        df.loc[isect[i], isect[i]] = val

    if copy:
        return df
    else:
        return len(isect)


def join_pandas_index(
    *args,
    method='union'
):
    """
    Join together an arbitrary number of pandas indices

    :param *args: Pandas indices or None
    :type *args: pd.Index, None
    :param method: Union or intersection join,
        defaults to 'union'
    :type method: str, optional
    :returns: One pandas index joined with the method chosen
    :rtype: pd.Index
    """

    idxs = [a for a in args if a is not None]

    if len(idxs) == 0:
        return None

    elif len(idxs) == 1:
        return idxs[0]

    elif method == 'intersection':
        return functools.reduce(
            lambda x, y: x.intersection(y),
            idxs
        )

    elif method == 'union':
        return functools.reduce(
            lambda x, y: x.union(y),
            idxs
        )

    else:
        raise ValueError(
            'method must be "union" or "intersection"'
        )


def align_dataframe_fill(
    df,
    index=None,
    columns=None,
    fillna=None
):
    """
    Align a dataframe and fill any NAs

    :param df: DataFrame to align
    :type df: pd.DataFrame
    :param index: Index, defaults to None
    :type index: pd.Index, optional
    :param columns: Columns, defaults to None
    :type columns: pd.Index, optional
    :param fillna: Fill value, defaults to None
    :type fillna: any, optional
    :return: Aligned dataframe
    :rtype: pd.DataFrame
    """

    if index is not None:
        df = df.reindex(
            index, axis=0
        )

    if columns is not None:
        df = df.reindex(
            columns, axis=1
        )

    if fillna is not None:
        df = df.fillna(fillna)

    return df


def array_set_diag(
    arr,
    val,
    row_labels,
    col_labels
):
    """
    Sets the diagonal of an 2D array to a value.
    Diagonal in this case is anything where row label == column label.

    :param arr: Array to modify in place
    :type arr: np.ndarray
    :param val: Value to insert into any cells where
        row label == column label
    :type val: numeric
    :param row_labels: Labels which correspond to the rows
    :type row_labels: list, pd.Index
    :param col_labels: Labels which correspond to the columns
    :type col_labels: list, pd.Index
    :return: Return the number of common row and column labels
    :rtype: int
    """

    if arr.ndim != 2:
        raise ValueError("Array must be 2D")

    # Find all the labels that are shared between rows and columns
    isect = set(row_labels).intersection(col_labels)

    # Set the value where row and column names are the same
    for i in isect:
        arr[row_labels == i, col_labels == i] = val

    return len(isect)


def make_array_2d(arr):
    """
    Changes array shape from 1d to 2d if needed (in-place)
    :param arr:  np.ndarray
    """
    if arr.ndim == 1:
        arr.shape = (arr.shape[0], 1)


def melt_and_reindex_dataframe(
    data_frame,
    value_name,
    idx_name="target",
    col_name="regulator"
):
    """
    Take a pandas dataframe and melt it into a one column dataframe
        (with the column `value_name`) and a multiindex
    of the original index + column
    :param data_frame: pd.DataFrame [M x N]
        Meltable dataframe
    :param value_name: str
        The column name for the values of the dataframe
    :param idx_name: str
        The name to assign to the original data_frame index values
    :param col_name: str
        The name to assign to the original data_frame column values
    :return: pd.DataFrame [(M*N) x 1]
        Melted dataframe with a single column of values and a multiindex
        that is the original index + column for that value
    """

    # Copy the dataframe and move the index to a column
    data_frame = data_frame.copy()
    data_frame[idx_name] = data_frame.index

    # Melt it into a [(M*N) x 3] dataframe
    data_frame = data_frame.melt(
        id_vars=idx_name,
        var_name=col_name,
        value_name=value_name
    )

    # Create a multiindex and then drop the columns
    # that are now in the index
    data_frame.index = pd.MultiIndex.from_frame(
        data_frame.loc[:, [idx_name, col_name]]
    )

    del data_frame[idx_name]
    del data_frame[col_name]

    return data_frame


def apply_window_vector(
    vec,
    window,
    func
):
    """
    Apply a function to a 1d array by windows.
    For logical testing of an array without having
    to allocate a full array of bools

    :param vec: A 1d vector to be normalized
    :type vec: np.ndarray, sp.sparse.spmatrix
    :param window: The window size to process
    :type window: int
    :param func: A function that produces an aggregate
        result for the window
    :type func: callable
    :return:
    """

    return np.array([
        func(vec[i * window:min((i + 1) * window, len(vec))])
        for i in range(int(np.ceil(len(vec) / window)))
    ])


def safe_apply_to_array(
    array,
    func,
    *args,
    axis=0,
    dtype=None,
    **kwargs
):
    """
    Applies a function to a 2d array
    Safe for both sparse and dense
    """

    if sparse.issparse(array):

        if dtype is None:
            dtype = array.dtype

        out_arr = np.empty(array.shape, dtype=dtype)

        if axis == 0:
            for i in range(array.shape[1]):
                out_arr[:, i] = func(
                    array[:, i].A.ravel(),
                    *args,
                    **kwargs
                )

        elif axis == 1:
            for i in range(array.shape[0]):
                out_arr[i, :] = func(
                    array[i, :].A.ravel(),
                    *args,
                    **kwargs
                )

        return out_arr

    else:
        return np.apply_along_axis(
            func,
            axis,
            array,
            *args,
            **kwargs
        )


def is_array_float(
    array
):
    """
    Is the dtype of an array a float

    :param array: Array
    :type array: np.ndarray
    :return: Is a float dtype
    :rtype: bool
    """

    return pat.is_float_dtype(array.dtype)


def convert_array_to_float(
    array,
    inplace=True
):
    """
    Convert an array to floats

    :param array: _description_
    :type array: _type_
    :param inplace: _description_, defaults to True
    :type inplace: bool, optional
    """

    # Return as-is if it's floats already
    if is_array_float(array):
        return array

    # Make a copy if inplace is False
    elif not inplace:
        return array.astype(np.float64)

    # If it's a compatible-width integer dtype
    # convert in-place and return
    # Otherwise return a copy
    if array.dtype == np.int32:
        dtype = np.float32
    elif array.dtype == np.int64:
        dtype = np.float64
    else:
        return array.astype(np.float64)

    # Create a new memoryview with a specific dtype
    float_view = array.view(dtype)

    # Assign the old data through the memoryview
    float_view[:] = array

    # Replace the old data with the newly converted data
    return float_view
