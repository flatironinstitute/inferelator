from __future__ import print_function, unicode_literals, division

import copy as cp
import gc
import math
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import scipy.stats
import pandas.api.types as pat
import scipy.io
from anndata import AnnData
from inferelator.utils import Debug, Validator


def dot_product(a, b, dense=True, cast=True):
    """
    Dot product two matrices together. Allow either matrix (or both or neither) to be sparse.

    :param a:
    :param b:
    :param dense:
    :param cast:
    :return:
    """
    if sparse.isspmatrix(a) and sparse.isspmatrix(b):
        return a.dot(b).A if dense else a.dot(b)
    elif sparse.isspmatrix(a):
        return a.dot(sparse.csr_matrix(b)).A if dense else a.dot(sparse.csr_matrix(b))
    elif sparse.isspmatrix(b):
        return np.dot(a, b.A)
    else:
        return np.dot(a, b)


class DotProduct:

    _dot_func = dot_product

    @classmethod
    def set_mkl(cls, mkl=True):

        # If the MKL flag is None, don't change anything
        if mkl is None:
            pass

        # If the MKL flag is True, use the dot_product_mkl function when .dot() is called
        if mkl:
            try:
                from sparse_dot_mkl import get_version_string, dot_product_mkl as dp
                msg = "Matrix multiplication will use sparse_dot_mkl package with MKL: {m}"
                vstring = get_version_string()
                Debug.vprint(msg.format(m=vstring if vstring is not None else "Install mkl-service for details"),
                             level=2)

                cls._dot_func = dp

            # If it isn't available, use the scipy/numpy functions instead
            except ImportError as err:
                Debug.vprint("Unable to load MKL with sparse_dot_mkl:\n" + str(err), level=0)
                cls._dot_func = dot_product

        # If the MKL flag is True, use the python (numpy/scipy) functions when .dot() is called
        else:
            Debug.vprint("Matrix multiplication will use Numpy; this is not advised for sparse data", level=2)
            cls._dot_func = dot_product

    @classmethod
    def dot(cls, *args, **kwargs):
        return cls._dot_func(*args, **kwargs)


def df_from_tsv(file_like, has_index=True):
    "Read a tsv file or buffer with headers and row ids into a pandas dataframe."
    return pd.read_csv(file_like, sep="\t", header=0, index_col=0 if has_index else False)


def df_set_diag(df, val, copy=True):
    """
    Sets the diagonal of a dataframe to a value. Diagonal in this case is anything where row label == column label.

    :param df: pd.DataFrame
        DataFrame to modify
    :param val: numeric
        Value to insert into any cells where row label == column label
    :param copy: bool
        Force-copy the dataframe instead of modifying in place
    :return: pd.DataFrame / int
        Return either the modified dataframe (if copied) or the number of cells modified (if changed in-place)
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


def array_set_diag(arr, val, row_labels, col_labels):
    """
    Sets the diagonal of an 2D array to a value. Diagonal in this case is anything where row label == column label.

    :param arr: Array to modify in place
    :type arr: np.ndarray
    :param val: Value to insert into any cells where row label == column label
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


def melt_and_reindex_dataframe(data_frame, value_name, idx_name="target", col_name="regulator"):
    """
    Take a pandas dataframe and melt it into a one column dataframe (with the column `value_name`) and a multiindex
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
        Melted dataframe with a single column of values and a multiindex that is the original index + column for
        that value
    """

    # Copy the dataframe and move the index to a column
    data_frame = data_frame.copy()
    data_frame[idx_name] = data_frame.index

    # Melt it into a [(M*N) x 3] dataframe
    data_frame = data_frame.melt(id_vars=idx_name, var_name=col_name, value_name=value_name)

    # Create a multiindex and then drop the columns that are now in the index
    data_frame.index = pd.MultiIndex.from_frame(data_frame.loc[:, [idx_name, col_name]])
    del data_frame[idx_name]
    del data_frame[col_name]

    return data_frame


def scale_vector(vec, ddof=1):
    """
    Take a vector and normalize it to a mean 0 and standard deviation 1 (z-score)

    :param vec: A 1d vector to be normalized
    :type vec: np.ndarray, sp.sparse.spmatrix
    :param ddof: The delta degrees of freedom for variance calculation
    :type ddof: int
    :return: A centered and scaled vector
    :rtype: np.ndarray
    """

    # Convert a sparse vector to a dense vector
    if sparse.isspmatrix(vec):
        vec = vec.A

    # Return 0s if the variance is 0
    if np.var(vec) == 0:
        return np.zeros(vec.shape, dtype=float)

    # Otherwise scale with scipy.stats.zscore
    else:
        return scipy.stats.zscore(vec, axis=None, ddof=ddof)


def apply_window_vector(vec, window, func):
    """
    Apply a function to a 1d array by windows.
    For logical testing of an array without having to allocate a full array of bools

    :param vec: A 1d vector to be normalized
    :type vec: np.ndarray, sp.sparse.spmatrix
    :param window: The window size to process
    :type window: int
    :param func: A function that produces an aggregate result for the window
    :type func: callable
    :return:
    """
    steps = math.ceil(len(vec) / window)
    return np.array([func(vec[i * window:min((i + 1) * window, len(vec))]) for i in range(steps)])


class InferelatorData(object):
    """ Store inferelator data in an AnnData object. This will always be Samples by Genes """

    name = None

    _adata = None

    @property
    def _is_integer(self):
        return pat.is_integer_dtype(self._adata.X.dtype)

    @property
    def expression_data(self):
        return self._adata.X

    @property
    def values(self):
        return self._adata.X

    @property
    def _data(self):
        if self.is_sparse:
            return self._adata.X.data
        else:
            return self._adata.X

    @_data.setter
    def _data(self, new_data):
        if self.is_sparse:
            self._adata.X.data = new_data
        else:
            self._adata.X = new_data

    @property
    def _data_mem_usage(self):
        if self.is_sparse:
            return self._adata.X.data.nbytes + self._adata.X.indices.nbytes + self._adata.X.indptr.nbytes
        else:
            return self._adata.X.nbytes

    @property
    def meta_data(self):
        return self._adata.obs

    @meta_data.setter
    def meta_data(self, new_meta_data):

        if isinstance(new_meta_data, InferelatorData):
            new_meta_data = new_meta_data.meta_data

        # Reindex the new metadata to match the existing sample names
        new_meta_data = new_meta_data.copy()
        new_meta_data.index = new_meta_data.index.astype(str)

        # Force unique names by appending values
        if self._adata.obs_names.nunique() != self.num_obs:
            self._adata.obs_names_make_unique()

        # Drop duplicate names on the new meta data
        if new_meta_data.index.nunique() != new_meta_data.shape[0]:
            new_meta_data = new_meta_data.loc[~new_meta_data.duplicated(), :]

        # If the new one is the right size, force it in one way or the other
        # Reindex the metadata to match the sample names
        try:
            new_meta_data = new_meta_data.reindex(self.sample_names)
        except ValueError:
            # If the metadata is the wrong size, angrily die
            if new_meta_data.shape[0] != self.num_obs:
                msg = "Metadata size {sh1} does not match data ({sh2})".format(sh1=new_meta_data.shape,
                                                                                          sh2=self.num_obs)
                raise ValueError(msg)
            new_meta_data.index = self.sample_names

        if len(self._adata.obs.columns) > 0:
            keep_columns = self._adata.obs.columns.difference(new_meta_data.columns)
            self._adata.obs = pd.concat((new_meta_data, self._adata.obs.loc[:, keep_columns]), axis=1)
        else:
            self._adata.obs = new_meta_data

    @property
    def gene_data(self):
        return self._adata.var

    @gene_data.setter
    def gene_data(self, new_gene_data):

        if isinstance(new_gene_data, InferelatorData):
            new_gene_data = new_gene_data.gene_data

        new_gene_data = new_gene_data.copy()
        new_gene_data.index = new_gene_data.index.astype(str)
        # Use the intersection of this and the expression data genes to make a list of gene names to keep
        self._adata.uns["trim_gene_list"] = new_gene_data.index.intersection(self._adata.var.index)

        new_gene_data = new_gene_data.reindex(self._adata.var_names)

        # Join any new columns to any existing columns
        # Update (overwrite) any columns in the existing meta data if they are in the new meta data
        if len(self._adata.var.columns) > 0:
            keep_columns = self._adata.var.columns.difference(new_gene_data.columns)
            self._adata.var = pd.concat((new_gene_data, self._adata.var.loc[:, keep_columns]), axis=1)
        else:
            self._adata.var = new_gene_data

    @property
    def gene_names(self):
        return self._adata.var_names

    @property
    def gene_counts(self):
        return self._adata.X.sum(axis=0).A.flatten() if self.is_sparse else self._adata.X.sum(axis=0)

    @property
    def sample_names(self):
        return self._adata.obs_names

    @property
    def sample_counts(self):
        return self._adata.X.sum(axis=1).A.flatten() if self.is_sparse else self._adata.X.sum(axis=1)

    @property
    def sample_means(self):
        return self._adata.X.mean(axis=1).A.flatten() if self.is_sparse else self._adata.X.mean(axis=1)

    @property
    def sample_stdev(self):
        return self._adata.X.std(axis=1, ddof=1).A.flatten() if self.is_sparse else self._adata.X.std(axis=1, ddof=1)

    @property
    def non_finite(self):
        if min(self._data.shape) == 0:
            return 0, None
        elif self.is_sparse:
            nnf = np.sum(apply_window_vector(self._adata.X.data, 1000000, lambda x: np.sum(~np.isfinite(x))))
            return nnf, ["GENES_NOT_ID_SPARSE_MATRIX"] if nnf > 0 else None
        else:
            non_finite = np.apply_along_axis(lambda x: np.sum(~np.isfinite(x)) > 0, 0, self._data)
            nnf = np.sum(non_finite)
            return nnf, self.gene_names[non_finite] if nnf > 0 else None

    @property
    def is_sparse(self):
        return sparse.issparse(self._adata.X)

    @property
    def shape(self):
        return self._adata.shape

    @property
    def num_obs(self):
        return self._adata.shape[0]

    @property
    def num_genes(self):
        return self._adata.shape[1]

    def __getattr__(self, item):
        return getattr(self._adata, item)

    def __str__(self):
        msg = "InferelatorData [{dt} {sh}, Metadata {me}] Memory: {mem:.2f} MB"
        return msg.format(sh=self.shape, dt=self._data.dtype, me=self.meta_data.shape,
                          mem=(self._data_mem_usage / 1e6))

    def __init__(self, expression_data=None, transpose_expression=False, meta_data=None, gene_data=None,
                 gene_data_idx_column=None, gene_names=None, sample_names=None, dtype=None, name=None):
        """
        Create a new InferelatorData object

        :param expression_data: A tabular observations x variables matrix
        :type expression_data: np.array, scipy.sparse.spmatrix, anndata.AnnData, pd.DataFrame
        :param transpose_expression: Should the table be transposed. Defaults to False
        :type transpose_expression: bool
        :param meta_data: Meta data for observations. Needs to align to the expression data
        :type meta_data: pd.DataFrame
        :param gene_data: Meta data for variables.
        :type gene_data: pd.DataFrame
        :param gene_data_idx_column: The gene_data column which should be used to align to expression_data
        :type gene_data_idx_column: bool
        :param gene_names: Names to be used for variables.
            Will be inferred from a dataframe or anndata object if not set.
        :type gene_names: list, pd.Index
        :param sample_names: Names to be used for observations
            Will be inferred from a dataframe or anndata object if not set.
        :type sample_names: list, pd.Index
        :param dtype: Explicitly convert the data to this dtype if set.
            Only applies to data loaded from a pandas dataframe. Numpy arrays and scipy matrices will use existing type.
        :type dtype: np.dtype
        :param name: Name of this data structure.
        :type name: None, str
        """

        if expression_data is not None and isinstance(expression_data, pd.DataFrame):
            object_cols = expression_data.dtypes == object

            if sum(object_cols) > 0:
                object_data = expression_data.loc[:, object_cols]
                meta_data = object_data if meta_data is None else pd.concat((meta_data, object_data), axis=1)
                expression_data.drop(expression_data.columns[object_cols], inplace=True, axis=1)

            if dtype is None and all(map(lambda x: pat.is_integer_dtype(x), expression_data.dtypes)):
                dtype = 'int32'
            elif dtype is None:
                dtype = 'float64'

            self._make_idx_str(expression_data)

            if transpose_expression:
                self._adata = AnnData(X=expression_data.T, dtype=dtype)
            else:
                self._adata = AnnData(X=expression_data, dtype=dtype)

        elif expression_data is not None and isinstance(expression_data, AnnData):
            self._adata = expression_data

        elif expression_data is not None:
            self._adata = AnnData(X=expression_data.T if transpose_expression else expression_data,
                                  dtype=expression_data.dtype)

        else:
            self._adata = AnnData()

        if gene_names is not None and len(gene_names) > 0:
            self._adata.var_names = gene_names

        if sample_names is not None and len(sample_names) > 0:
            self._adata.obs_names = sample_names

        if meta_data is not None:
            self._make_idx_str(meta_data)
            self.meta_data = meta_data

        if gene_data is not None:
            if gene_data_idx_column is not None and gene_data_idx_column in gene_data:
                gene_data.index = gene_data[gene_data_idx_column]
            elif gene_data_idx_column is not None:
                msg = "No gene_data column {c} in {a}".format(c=gene_data_idx_column, a=" ".join(gene_data.columns))
                raise ValueError(msg)
            self._make_idx_str(gene_data)
            self.gene_data = gene_data

        self._cached = {}
        self.name = name

    def convert_to_float(self):
        """
        Convert the data in-place to a float dtype
        """

        if pat.is_float_dtype(self._data.dtype):
            return None
        elif self._data.dtype == np.int32:
            dtype = np.float32
        elif self._data.dtype == np.int64:
            dtype = np.float64
        else:
            raise ValueError("Data is not float, int32, or int64")

        # Create a new memoryview with a specific dtype
        float_view = self._data.view(dtype)

        # Assign the old data through the memoryview
        float_view[:] = self._data

        # Replace the old data with the newly converted data
        self._data = float_view

    def trim_genes(self, remove_constant_genes=True, trim_gene_list=None):
        """
        Remove genes (columns) that are unwanted from the data set. Do this in-place.

        :param remove_constant_genes:
        :type remove_constant_genes: bool
        :param trim_gene_list: This is a list of genes to KEEP.
        :type trim_gene_list: list, pd.Series, pd.Index
        """

        keep_column_bool = np.ones((len(self._adata.var_names),), dtype=bool)

        if trim_gene_list is not None:
            keep_column_bool &= self._adata.var_names.isin(trim_gene_list)
        if "trim_gene_list" in self._adata.uns:
            keep_column_bool &= self._adata.var_names.isin(self._adata.uns["trim_gene_list"])

        list_trim = len(self._adata.var_names) - np.sum(keep_column_bool)
        comp = 0 if self._is_integer else np.finfo(self.values.dtype).eps * 10

        if remove_constant_genes:
            nz_var = (self.values.max(axis=0) - self.values.min(axis=0))
            nz_var = comp < (nz_var.A.flatten() if self.is_sparse else nz_var)

            keep_column_bool &= nz_var
            var_zero_trim = np.sum(nz_var)
        else:
            var_zero_trim = 0

        if np.sum(keep_column_bool) == 0:
            err_msg = "No genes remain after trimming. ({lst} removed to match list, {v} removed for var=0)"
            raise ValueError(err_msg.format(lst=list_trim, v=var_zero_trim))

        if np.sum(keep_column_bool) == self._adata.shape[1]:
            pass
        else:
            Debug.vprint("Trimming {name} matrix {sh} to {n} columns".format(name=self.name, sh=self._adata.X.shape,
                                                                             n=np.sum(keep_column_bool)),
                         level=1)

            # This explicit copy allows the original to be deallocated
            # Otherwise the GC leaves the original because the view reference keeps it alive
            # At some point it will need to copy so why not now
            self._adata = AnnData(self._adata.X[:, keep_column_bool],
                                  obs=self._adata.obs.copy(),
                                  var=self._adata.var.loc[keep_column_bool, :].copy(),
                                  dtype=self._adata.X.dtype)

            # Make sure that there's no hanging reference to the original object
            gc.collect()

    def get_gene_data(self, gene_list, copy=False, force_dense=False, to_df=False, zscore=False):

        x = self._adata[:, gene_list]
        labels = x.var_names

        if (force_dense or to_df or zscore) and self.is_sparse:
            x = x.X.A
        else:
            x = x.X

        if zscore:
            x = np.subtract(x, self.obs_means.reshape(-1, 1))
            x = np.divide(x, self.obs_stdev.reshape(-1, 1))
        elif copy:
            x = x.copy()

        return pd.DataFrame(x, columns=labels, index=self.sample_names) if to_df else x

    def get_sample_data(self, sample_index, copy=False, force_dense=False, to_df=False, zscore=False):

        x = self._adata[sample_index, :]
        labels = x.obs_names

        if (force_dense or to_df or zscore) and self.is_sparse:
            x = x.X.A
        else:
            x = x.X

        if zscore:
            x = np.subtract(x, self.obs_means[sample_index].reshape(-1, 1))
            x = np.divide(x, self.obs_stdev[sample_index].reshape(-1, 1))
        elif copy:
            x = x.X.copy()

        return pd.DataFrame(x, columns=self.gene_names, index=labels) if to_df else x

    def get_bootstrap(self, sample_bootstrap_index):
        return InferelatorData(expression_data=self._adata[sample_bootstrap_index, :].X.copy(),
                               gene_names=self.gene_names)

    def subset_copy(self, row_index=None, column_index=None):

        if row_index is not None and column_index is not None:
            data_view = self._adata[row_index, column_index]
        elif row_index is not None:
            data_view = self._adata[row_index, :]
        elif column_index is not None:
            data_view = self._adata[: column_index]
        else:
            data_view = self._adata

        return InferelatorData(data_view.copy())

    def dot(self, other, other_is_right_side=True, force_dense=False):
        """
        Calculate dot product
        :param other:
        :param other_is_right_side:
        :param force_dense:
        :return:
        """

        if other_is_right_side:
            return DotProduct.dot(self._adata.X, other, cast=True, dense=force_dense)
        else:
            return DotProduct.dot(other, self._adata.X, cast=True, dense=force_dense)

    def to_csv(self, file_name, sep="\t"):

        if self.is_sparse:
            scipy.io.mmwrite(file_name, self.values)
        else:
            self._adata.to_df().to_csv(file_name, sep=sep)

    def to_h5ad(self, file_name, compression="gzip"):

        self._adata.write(file_name, compression=compression)

    def transform(self, func, add_pseudocount=False, memory_efficient=True, chunksize=1000):

        if add_pseudocount and self.is_sparse:
            self._adata.X.data += 1
        elif add_pseudocount:
            self._adata.X += 1

        if self.is_sparse:
            self._adata.X.data = func(self._adata.X.data)
        elif self._adata.X.ndim == 1 or self._is_integer:
            self._adata.X = func(self._adata.X)
        elif not memory_efficient and type(func(self._data.flat[0])) == self._adata.X.dtype:
            self._adata.X[...] = func(self._adata.X)
        elif memory_efficient and type(func(self._data.flat[0])) == self._adata.X.dtype:
            for i in range(math.ceil(self._adata.shape[0] / chunksize)):
                start, stop = i * chunksize, min(i + 1 * chunksize, self._adata.shape[0])
                self._adata.X[start:stop, :] = func(self._adata.X[start:stop, :])
        else:
            self._adata.X = func(self._adata.X)

    def add(self, val):
        """
        Add a value to the matrix in-place
        :param val: Value to add
        :type val: numeric
        """
        self._data[...] = self._data + val

    def subtract(self, val):
        """
        Subtract a value from the matrix in-place
        :param val: Value to subtract
        :type val: numeric
        """
        self._data[...] = self._data - val

    def divide(self, div_val, axis=None):
        """
        Divide a value from the matrix in-place
        :param div_val: Value to divide
        :type div_val: numeric
        :param axis: Which axis to divide from (0, 1, or None)
        :type axis: int, None
        """

        if self._is_integer:
            self.convert_to_float()

        if self.is_sparse and axis is None:
            self._adata.X.data /= div_val
        elif self.is_sparse and ((sparse.isspmatrix_csr(self._adata.X) and axis == 1) or
                                 (sparse.isspmatrix_csc(self._adata.X) and axis == 0)):
            if not hasattr(div_val, "ndim") or div_val.ndim != 1 or self.shape[0 if axis else 1] != div_val.shape[0]:
                raise ValueError("Division array is not aligned")
            self._adata.X.data /= np.repeat(div_val, self._adata.X.getnnz(axis=axis))
        elif self.is_sparse:
            raise ValueError("axis = 1 only works for CSC & axis = 0 only works for CSR")
        elif axis is None:
            self._adata.X /= div_val
        elif axis == 0:
            self._adata.X /= div_val[None, :]
        elif axis == 1:
            self._adata.X /= div_val[:, None]
        else:
            raise ValueError("axis must be 0, 1 or None")

    def multiply(self, mult_val, axis=None):
        """
        Multiply the matrix by a value in-place
        :param mult_val: Value to multiply
        :type mult_val: numeric
        :param axis: Which axis to divide from (0, 1, or None)
        :type axis: int, None
        """

        if self._is_integer:
            self.convert_to_float()

        if self.is_sparse and axis is None:
            self._adata.X.data *= mult_val
        elif self.is_sparse and ((sparse.isspmatrix_csr(self._adata.X) and axis == 1) or
                                 (sparse.isspmatrix_csc(self._adata.X) and axis == 0)):
            if not hasattr(mult_val, "ndim") or mult_val.ndim != 1 or self.shape[0 if axis else 1] != mult_val.shape[0]:
                raise ValueError("Division array is not aligned")
            self._adata.X.data *= np.repeat(mult_val, self._adata.X.getnnz(axis=axis))
        elif self.is_sparse:
            raise ValueError("axis = 1 only works for CSC & axis = 0 only works for CSR")
        elif axis is None:
            self._adata.X *= mult_val
        elif axis == 0:
            self._adata.X *= mult_val[None, :]
        elif axis == 1:
            self._adata.X *= mult_val[:, None]
        else:
            raise ValueError("axis must be 0, 1 or None")

    def zscore(self, axis=0, ddof=1):

        self.convert_to_float()
        self.to_dense()

        if axis == 0:
            for i in range(self.shape[1]):
                self._data[:, i] = scale_vector(self._data[:, i], ddof=ddof)
        elif axis == 1:
            for i in range(self.shape[0]):
                self._data[i, :] = scale_vector(self._data[i, :], ddof=ddof)

        return self

    def copy(self):

        new_data = InferelatorData(self.values.copy(),
                                   meta_data=self.meta_data.copy(),
                                   gene_data=self.gene_data.copy())

        new_data._adata.var_names = cp.copy(self._adata.var_names)
        new_data._adata.obs_names = cp.copy(self._adata.obs_names)
        new_data._adata.uns = cp.copy(self._adata.uns)

        return new_data

    def to_csc(self):

        if self.is_sparse and not sparse.isspmatrix_csc(self._adata.X):
            self._adata.X = sparse.csc_matrix(self._adata.X)

    def to_csr(self):

        if self.is_sparse and not sparse.isspmatrix_csr(self._adata.X):
            self._adata.X = sparse.csr_matrix(self._adata.X)

    def to_dense(self):

        if self.is_sparse:
            self._adata.X = self._adata.X.A

    def to_sparse(self, mode="csr"):

        if not self.is_sparse and mode.lower() == "csr":
            self._adata.X = sparse.csr_matrix(self._adata.X)
        elif not self.is_sparse and mode.lower() == "csc":
            self._adata.X = sparse.csc_matrix(self._adata.X)
        elif not self.is_sparse:
            raise ValueError("Mode must be csc or csr")

    @staticmethod
    def _make_idx_str(df):
        df.index = df.index.astype(str) if not pat.is_string_dtype(df.index.dtype) else df.index
        df.columns = df.columns.astype(str) if not pat.is_string_dtype(df.columns.dtype) else df.columns
