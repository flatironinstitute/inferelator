from __future__ import print_function, unicode_literals, division

import pandas as pd
import numpy as np
import scipy.sparse as sparse
import pandas.api.types as pat
from anndata import AnnData
from inferelator.utils.debug import Debug


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


class InferelatorData(object):
    """ Store inferelator data in an AnnData object. This will always be Samples by Genes """

    _adata = None
    _is_integer = False

    @property
    def expression_data(self):
        if self._adata is not None:
            return self._adata.X
        else:
            return None

    @property
    def data(self):
        if self.is_sparse:
            return self._adata.X.data
        else:
            return self._adata.X

    @property
    def meta_data(self):
        if self._adata is not None:
            return self._adata.obs
        else:
            return None

    @meta_data.setter
    def meta_data(self, new_meta_data):
        new_meta_data.index = self._adata.obs.index
        if len(self._adata.obs.columns) > 0:
            keep_columns = self._adata.obs.columns.difference(new_meta_data.columns)
            self._adata.obs = pd.concat((new_meta_data, self._adata.obs.loc[:, keep_columns]))
        else:
            self._adata.obs = new_meta_data

    @property
    def gene_data(self):
        if self._adata is not None:
            return self._adata.var
        else:
            return None

    @gene_data.setter
    def gene_data(self, new_gene_data):
        self._adata.uns["trim_gene_list"] = new_gene_data.index.intersection(self._adata.var.index)
        self._adata.var = self._adata.var.join(new_gene_data)

    @property
    def gene_names(self):
        if self._adata is not None:
            return self._adata.var.index.astype(str)
        else:
            return None

    @property
    def sample_names(self):
        if self._adata is not None:
            return self._adata.obs.index.astype(str)
        else:
            return None

    @property
    def non_finite(self):
        if self.data.ndim == 1:
            return np.sum(~np.isfinite(self.data)), None
        elif min(self.data.shape) == 0:
            return 0, None
        else:
            non_finite = np.apply_along_axis(lambda x: np.sum(~np.isfinite(x)) > 0, 1, self.data)
            nnf = np.sum(non_finite)
            return nnf, self.gene_names[non_finite] if nnf > 0 else None

    @property
    def is_sparse(self):
        if self._adata is not None:
            return sparse.issparse(self._adata.X)
        else:
            return None

    @property
    def shape(self):
        if self._adata is not None:
            return self._adata.shape
        else:
            return None

    def __init__(self, expression_data, transpose_expression=False, meta_data=None, gene_data=None, gene_names=None,
                 sample_names=None):

        if isinstance(expression_data, pd.DataFrame):
            object_cols = expression_data.dtypes == object

            if sum(object_cols) > 0:
                object_data = expression_data.loc[:, object_cols]
                meta_data = object_data if meta_data is None else pd.concat((meta_data, object_data))
                expression_data.drop(expression_data.columns[object_cols], inplace=True, axis=1)

            if all(map(lambda x: pat.is_integer_dtype(x), expression_data.dtypes)):
                dtype = 'int32'
                self._is_integer = True
            else:
                dtype = 'float64'
                self._is_integer = False

            self._make_idx_str(expression_data)

            if transpose_expression:
                self._adata = AnnData(X=expression_data.T, dtype=dtype)
            else:
                self._adata = AnnData(X=expression_data, dtype=dtype)
        else:
            if transpose_expression:
                self._adata = AnnData(X=expression_data.T, dtype=expression_data.dtype)
            else:
                self._adata = AnnData(X=expression_data, dtype=expression_data.dtype)

            if gene_names is not None:
                self._adata.var_names = gene_names
            if sample_names is not None:
                self._adata.obs_names = sample_names

        if meta_data is not None:
            self._make_idx_str(meta_data)
            self._adata.obs = meta_data

        if gene_data is not None:
            self._make_idx_str(gene_data)
            self._adata = gene_data

    def trim_genes(self, remove_constant_genes=True, trim_gene_list=None):
        """
        Remove genes (columns) that are unwanted from the data set
        :param remove_constant_genes:
        :type remove_constant_genes: bool
        :param trim_gene_list:
        :return:
        """

        if trim_gene_list is not None:
            keep_column_bool = self._adata.var.index.isin(trim_gene_list)
        elif "trim_gene_list" in self._adata.uns:
            keep_column_bool = self._adata.var.index.isin(self._adata.uns["trim_gene_list"])
        else:
            keep_column_bool = np.ones((len(self._adata.var.index),), dtype=bool)

        list_trim = len(self._adata.var.index) - np.sum(keep_column_bool)

        if remove_constant_genes:
            if self.is_sparse:
                keep_column_bool &= self.expression_data.getnnz(axis=1) > 0
            else:
                comp = 0 if self._is_integer else np.finfo(self.expression_data.dtype).eps * 10
                keep_column_bool &= np.apply_along_axis(lambda x: np.max(x) - np.min(x), 0, self.expression_data) > comp

            var_zero_trim = len(self._adata.var.index) - np.sum(keep_column_bool) + list_trim
        else:
            var_zero_trim = 0

        if np.sum(keep_column_bool) == 0:
            err_msg = "No genes remain after trimming. ({lst} removed to match list, {v} removed for var=0)"
            raise ValueError(err_msg.format(lst=list_trim, v=var_zero_trim))

        if np.sum(keep_column_bool) == self._adata.shape[1]:
            pass
        else:
            # This explicit copy allows the original to be deallocated
            # Otherwise the GC leaves the original because the view reference keeps it alive
            # At some point it will need to copy so why not now
            self._adata = self._adata[:, keep_column_bool].copy()

    def get_genes(self, gene_list, copy=False):

        return self._adata[:, gene_list] if not copy else self._adata[:, gene_list].copy()

    def dot(self, other, other_is_right_side=True, force_dense=False):

        if self.is_sparse and sparse.issparse(other):
            dot_product = self._adata.X.dot(other) if other_is_right_side else other.dot(self._adata.X)
        elif self.is_sparse and not sparse.issparse(other) and other_is_right_side:
            other = sparse.csr_matrix(other)
            dot_product = self._adata.X.dot(other) if other_is_right_side else other.dot(self._adata.X)
        elif not self.is_sparse and sparse.issparse(other):
            dot_product = np.dot(self._adata.X, other.A) if other_is_right_side else np.dot(other.A, self._adata.X)
        else:
            dot_product = np.dot(self._adata.X, other) if other_is_right_side else np.dot(other, self._adata.X)

        if force_dense and sparse.issparse(dot_product):
            return dot_product.A
        else:
            return dot_product

    def to_csv(self, file_name):
        if self.is_sparse:
            Debug.vprint("Saving sparse arrays to text files is not supported", level=0)
        else:
            np.savetxt(file_name, self.expression_data, delimiter="\t", header="\t".join(self.gene_names))

    def transform(self, func, add_pseudocount=False):

        if add_pseudocount and self.is_sparse:
            self._adata.X.data += 1
        elif add_pseudocount:
            self._adata.X += 1

        if self.is_sparse:
            self._adata.X.data = func(self._adata.X.data)
        elif self._adata.X.ndim == 1 or self._is_integer:
            self._adata.X = func(self._adata.X)
        else:
            self._adata.X[...] = func(self._adata.X)

    @staticmethod
    def _make_idx_str(df):
        df.index = df.index.astype(str) if not pat.is_string_dtype(df.index.dtype) else df.index
        df.columns = df.columns.astype(str) if not pat.is_string_dtype(df.columns.dtype) else df.columns
