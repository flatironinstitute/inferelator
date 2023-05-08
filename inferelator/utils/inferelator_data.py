import gc
import pandas as pd
import numpy as np
import pandas.api.types as pat

from sklearn.preprocessing import StandardScaler
from anndata import AnnData

from scipy import (
    sparse,
    io
)

from inferelator.utils import (
    Debug,
    Validator as check
)

from inferelator.utils.data import (
    apply_window_vector,
    DotProduct,
    convert_array_to_float
)


class InferelatorData(object):
    """
    Store inferelator data in an AnnData object.
    This will always be Samples by Genes
    """

    name = None

    _adata = None

    @property
    def _is_integer(self):
        return pat.is_integer_dtype(self._adata.X.dtype)

    @property
    def expression_data(self):
        return self._adata.X

    @expression_data.setter
    def expression_data(self, new_data):
        self._adata.X = new_data

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
            return np.sum((
                self._adata.X.data.nbytes,
                self._adata.X.indices.nbytes,
                self._adata.X.indptr.nbytes
            ))
        else:
            return self._adata.X.nbytes

    @property
    def prior_data(self):

        if "prior_data" in self._adata.uns:
            return self._adata.uns["prior_data"]
        else:
            return None

    @prior_data.setter
    def prior_data(self, new_prior):
        self._adata.uns["prior_data"] = new_prior

    @property
    def tfa_prior_data(self):
        if "tfa_prior_data" in self._adata.uns:
            return self._adata.uns["tfa_prior_data"]
        else:
            return None

    @tfa_prior_data.setter
    def tfa_prior_data(self, new_prior):
        self._adata.uns["tfa_prior_data"] = new_prior

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
                raise ValueError(
                    f"Metadata size {new_meta_data.shape} "
                    f"does not match data ({self.num_obs})"
                )

            new_meta_data.index = self.sample_names

        if len(self._adata.obs.columns) > 0:
            keep_columns = self._adata.obs.columns.difference(
                new_meta_data.columns
            )

            self._adata.obs = pd.concat(
                (new_meta_data, self._adata.obs.loc[:, keep_columns]),
                axis=1
            )

        else:
            self._adata.obs = new_meta_data

    @property
    def gene_data(self):
        return self._adata.var

    @gene_data.setter
    def gene_data(
        self,
        new_gene_data
    ):

        if isinstance(new_gene_data, InferelatorData):
            new_gene_data = new_gene_data.gene_data

        new_gene_data = new_gene_data.copy()
        new_gene_data.index = new_gene_data.index.astype(str)

        # Use the intersection of this and the expression data genes
        # to make a list of gene names to keep
        _in_gene_data = new_gene_data.index.intersection(self.gene_names)
        self._adata.uns["trim_gene_list"] = _in_gene_data

        # Reindex to align to the existing data
        new_gene_data = new_gene_data.reindex(self._adata.var_names)

        # Join any new columns to any existing columns
        # Update (overwrite) any columns in the existing meta data if
        # they are in the new meta data
        if len(self._adata.var.columns) > 0:
            cols = self._adata.var.columns.difference(new_gene_data.columns)
            self._adata.var = pd.concat(
                (new_gene_data, self._adata.var.loc[:, cols]),
                axis=1
            )
        else:
            self._adata.var = new_gene_data

    @property
    def gene_names(self):
        return self._adata.var_names

    @property
    def gene_counts(self):
        return self._counts(axis=0)

    @property
    def gene_means(self):
        return self._means(axis=0)

    @property
    def gene_stdev(self):
        return self._stds(axis=0)

    @property
    def sample_names(self):
        return self._adata.obs_names

    @property
    def sample_counts(self):
        return self._counts(axis=1)

    @property
    def sample_means(self):
        return self._means(axis=1)

    @property
    def sample_stdev(self):
        return self._stds(axis=1)

    @property
    def non_finite(self):
        """
        Check to see if any values are non-finite

        :return: Number of non-finite values and gene names
        :rtype: int, pd.Index
        """

        if min(self._data.shape) == 0:
            return 0, None

        elif self.is_sparse:
            nnf = np.sum(
                apply_window_vector(
                    self._adata.X.data,
                    1000000,
                    lambda x: np.sum(~np.isfinite(x))
                )
            )

            if nnf > 0:
                return nnf, ["Skipping gene check (Sparse matrix)"]
            else:
                return 0, None

        else:
            non_finite = np.apply_along_axis(
                lambda x: np.sum(~np.isfinite(x)) > 0,
                0,
                self._data
            )
            nnf = np.sum(non_finite)

            if nnf > 0:
                return nnf, self.gene_names[non_finite]
            else:
                return 0, None

    @property
    def is_sparse(self):
        return sparse.issparse(self._adata.X)

    @property
    def shape(self):
        return self._adata.shape

    @property
    def size(self):
        return self._adata.X.size

    @property
    def num_obs(self):
        return self.shape[0]

    @property
    def num_genes(self):
        return self.shape[1]

    @property
    def uns(self):
        return self._adata.uns

    def __str__(self):
        msg = "InferelatorData [{dt} {sh}, Metadata {me}] Memory: {mem:.2f} MB"
        return msg.format(
            sh=self.shape,
            dt=self._data.dtype,
            me=self.meta_data.shape,
            mem=(self._data_mem_usage / 1e6)
        )

    def __init__(
        self,
        expression_data=None,
        transpose_expression=False,
        meta_data=None,
        gene_data=None,
        gene_data_idx_column=None,
        gene_names=None,
        sample_names=None,
        dtype=None,
        name=None
    ):
        """
        Create a new InferelatorData object

        :param expression_data: A tabular observations x variables matrix
        :type expression_data: np.array, scipy.sparse.spmatrix,
            anndata.AnnData, pd.DataFrame
        :param transpose_expression: Should the table be transposed.
            Defaults to False
        :type transpose_expression: bool
        :param meta_data: Meta data for observations.
            Needs to align to the expression data
        :type meta_data: pd.DataFrame
        :param gene_data: Meta data for variables.
        :type gene_data: pd.DataFrame
        :param gene_data_idx_column: The gene_data column which
            should be used to align to expression_data
        :type gene_data_idx_column: bool
        :param gene_names: Names to be used for variables.
            Will be inferred from a dataframe or anndata object if not set.
        :type gene_names: list, pd.Index
        :param sample_names: Names to be used for observations
            Will be inferred from a dataframe or anndata object if not set.
        :type sample_names: list, pd.Index
        :param dtype: Explicitly convert the data to this dtype if set.
            Only applies to data loaded from a pandas dataframe.
            Numpy arrays and scipy matrices will use existing type.
        :type dtype: np.dtype
        :param name: Name of this data structure.
        :type name: None, str
        """

        # Empty anndata object
        if expression_data is None:
            self._adata = AnnData()

        # Convert a dataframe to an anndata object
        elif isinstance(expression_data, pd.DataFrame):
            object_cols = expression_data.dtypes == object

            # Pull out any object columns and use them as metadata
            if sum(object_cols) > 0:
                object_data = expression_data.loc[:, object_cols]

                if meta_data is None:
                    meta_data = object_data
                else:
                    meta_data = pd.concat((meta_data, object_data), axis=1)

                expression_data.drop(
                    expression_data.columns[object_cols],
                    inplace=True, axis=1
                )

            # If dtype is provided, use it
            if dtype is not None:
                pass

            # If all dtypes are integer use int32
            elif all(map(
                lambda x: pat.is_integer_dtype(x),
                expression_data.dtypes
            )):
                dtype = 'int32'

            # Otherwise use floats
            else:
                dtype = 'float64'

            self._make_idx_str(expression_data)

            if transpose_expression:
                self._adata = AnnData(
                    X=expression_data.T.values.astype(dtype)
                )
                self._adata.obs_names = expression_data.columns
                self._adata.var_names = expression_data.index

            else:
                self._adata = AnnData(
                    X=expression_data.values.astype(dtype)
                )
                self._adata.obs_names = expression_data.index
                self._adata.var_names = expression_data.columns

        # Use an anndata object that already exists
        elif isinstance(expression_data, AnnData):
            self._adata = expression_data

        # Convert a numpy array to an anndata object
        else:

            if transpose_expression:
                expression_data = expression_data.T

            self._adata = AnnData(
                X=expression_data
            )

        # Use gene_names as var_names
        if gene_names is not None and len(gene_names) > 0:
            self._adata.var_names = gene_names

        # Use sample_names as obs_names
        if sample_names is not None and len(sample_names) > 0:
            self._adata.obs_names = sample_names

        # Use meta_data as obs
        if meta_data is not None:
            self._make_idx_str(meta_data)
            self.meta_data = meta_data

        # Use gene_data as var
        if gene_data is not None:

            if gene_data_idx_column is not None:

                try:
                    gene_data.index = gene_data[gene_data_idx_column]

                except KeyError:
                    raise ValueError(
                        f"No gene_data column {gene_data_idx_column} "
                        f"in {' '.join(gene_data.columns)}"
                    )

            self._make_idx_str(gene_data)
            self.gene_data = gene_data

        self._cached = {}
        self.name = name

    def convert_to_float(self):
        """
        Convert the data in-place to a float dtype
        """

        self._data = convert_array_to_float(self._data)

    def trim_genes(self, remove_constant_genes=True, trim_gene_list=None):
        """
        Remove genes (columns) that are unwanted from the data set.
        Do this in-place.

        :param remove_constant_genes:
        :type remove_constant_genes: bool
        :param trim_gene_list: This is a list of genes to KEEP.
        :type trim_gene_list: list, pd.Series, pd.Index
        """

        keep_column_bool = np.ones((len(self._adata.var_names),), dtype=bool)

        if trim_gene_list is not None:
            keep_column_bool &= self._adata.var_names.isin(
                trim_gene_list
            )

        if "trim_gene_list" in self._adata.uns:
            keep_column_bool &= self._adata.var_names.isin(
                self._adata.uns["trim_gene_list"]
            )

        list_trim = len(self._adata.var_names) - np.sum(keep_column_bool)
        comp = 0 if self._is_integer else np.finfo(self.values.dtype).eps * 10

        if remove_constant_genes:
            nz_var = self.values.max(axis=0) - self.values.min(axis=0)
            nz_var = nz_var.A.flatten() if self.is_sparse else nz_var

            if np.any(np.isnan(nz_var)):
                raise ValueError(
                    f"NaN values are present in data matrix {self.name} "
                    f"when removing zero variance genes")

            nz_var = comp < nz_var

            keep_column_bool &= nz_var
            var_zero_trim = np.sum(nz_var)
        else:
            var_zero_trim = 0

        if np.sum(keep_column_bool) == 0:
            raise ValueError(
                "No genes remain after trimming. "
                f"({list_trim} removed to match list, "
                f"{var_zero_trim} removed for zero variance)"
            )

        if np.sum(keep_column_bool) == self._adata.shape[1]:
            pass
        else:
            Debug.vprint(
                f"Trimming {self.name} matrix "
                f"{self._adata.X.shape} to "
                f"{np.sum(keep_column_bool)} columns",
                level=1
            )

            # This explicit copy allows the original to be deallocated
            # Otherwise the view reference keeps it in memory
            # At some point it will need to copy so why not now
            self._adata = AnnData(
                self._adata.X[:, keep_column_bool],
                obs=self._adata.obs.copy(),
                var=self._adata.var.loc[keep_column_bool, :].copy()
            )

            # Make sure that there's no hanging reference to the original
            gc.collect()

    def get_gene_data(
        self,
        gene_list,
        force_dense=False,
        to_df=False,
        flatten=False
    ):

        x = self._adata[:, gene_list]
        labels = x.var_names

        if (force_dense or to_df) and self.is_sparse:
            x = x.X.A

        else:
            # Copy is necessary to get the numpy array
            # and not an anndata arrayview
            x = x.X.copy()

        if flatten:
            x = x.ravel()

        if to_df:
            return pd.DataFrame(
                x,
                columns=labels,
                index=self.sample_names
            )

        else:
            return x

    def get_sample_data(
        self,
        sample_index,
        copy=False,
        force_dense=False,
        to_df=False
    ):

        x = self._adata[sample_index, :]
        labels = x.obs_names

        if (force_dense or to_df) and self.is_sparse:
            x = x.X.A
        else:
            x = x.X

        if copy:
            x = x.X.copy()

        if to_df:
            x = pd.DataFrame(
                x,
                columns=self.gene_names,
                index=labels
            )

        return x

    def get_bootstrap(
        self,
        sample_bootstrap_index
    ):

        if sample_bootstrap_index is None:
            return self.copy()

        return InferelatorData(
            expression_data=self._adata.X[sample_bootstrap_index, :].copy(),
            gene_names=self.gene_names
        )

    def get_random_samples(
        self,
        num_obs,
        with_replacement=False,
        random_seed=None,
        random_gen=None,
        inplace=False,
        fix_names=True
    ):
        """
        Randomly sample to a specific number of observatons
        from the entire data set

        :param num_obs: Number of observations to return
        :type num_obs: int
        :param with_replacement: Sample with replacement, defaults to False
        :type with_replacement: bool, optional
        :param random_seed: Seed for numpy random generator, defaults to None.
            Will be ignored if a generator itself is passed to random_gen.
        :type random_seed: int, optional
        :param random_gen: Numpy random generator to use, defaults to None.
        :type random_gen: np.random.Generator, optional
        :param inplace: Change this instance of the data structure inplace
            and return a reference to itself
        :type inplace: bool, optional
        """

        check.argument_integer(num_obs, low=1)
        check.argument_integer(random_seed, allow_none=True)

        if (num_obs > self.num_obs) and not with_replacement:
            raise ValueError(
                f"Unable to sample {num_obs} from {self.num_obs} "
                "observations without replacement"
            )

        # Make a new random generator if not provided
        if random_gen is None:
            random_gen = np.random.default_rng(random_seed)

        # Sample with replacement using randint
        if with_replacement:
            keeper_ilocs = random_gen.integers(
                self.num_obs,
                size=(num_obs,)
            )

        # Sample without replacement using choice
        else:
            keeper_ilocs = random_gen.choice(
                np.arange(self.num_obs),
                size=(num_obs,),
                replace=False
            )

        # Subset AnnData object
        _new_adata = self._adata[keeper_ilocs, :]

        if _new_adata.is_view:
            _new_adata = _new_adata.copy()

        # Check names and make unique if set
        if with_replacement and fix_names:
            _new_adata.obs_names_make_unique()

        # Change this instance's _adata (explicit copy allows the old data to
        # be dereferenced instead of held as view)
        if inplace:
            self._adata = _new_adata
            return_obj = self
            gc.collect()

        # Create a new InferelatorData instance with the _adata slice
        else:
            return_obj = InferelatorData(_new_adata)

        return return_obj

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

    def dot(
        self,
        other,
        other_is_right_side=True,
        force_dense=False
    ):
        """
        Calculate dot product
        :param other:
        :param other_is_right_side:
        :param force_dense:
        :return:
        """

        if other_is_right_side:
            return DotProduct.dot(
                self._adata.X,
                other,
                cast=True,
                dense=force_dense
            )
        else:
            return DotProduct.dot(
                other,
                self._adata.X,
                cast=True,
                dense=force_dense
            )

    def to_csv(self, file_name, sep="\t"):

        if self.is_sparse:
            io.mmwrite(file_name, self.values)
        else:
            self._adata.to_df().to_csv(file_name, sep=sep)

    def to_h5ad(self, file_name, compression="gzip"):

        self._adata.write(file_name, compression=compression)

    def transform(
        self,
        func,
        add_pseudocount=False,
        memory_efficient=True,
        chunksize=1000
    ):
        """
        Apply a function to each nonzero value of a sparse matrix
        or each value of a dense matrix

        :param func: Function which takes a scalar or array input
        :type func: callable
        :param add_pseudocount: Add a pseudocount before applying function,
            defaults to False
        :type add_pseudocount: bool, optional
        :param memory_efficient: Apply function to chunks of the data and
            overwrite values in the existing array. Only works if the dtype
            returned from the function is the same as the dtype of the array.
            Defaults to True.
        :type memory_efficient: bool, optional
        :param chunksize: Number of observations to use as a chunk,
            defaults to 1000
        :type chunksize: int, optional
        """

        # Add 1 to every non-zero value
        if add_pseudocount and self.is_sparse:
            self._adata.X.data += 1
        elif add_pseudocount:
            self._adata.X += 1

        _type_match = type(func(self._data.flat[0])) == self._data.dtype

        # Apply function to the data if it's sparse
        if self.is_sparse:
            self._data = func(self._data)

        # Apply function to the data
        # by making a new data object
        elif self._adata.X.ndim == 1 or self._is_integer:
            self._data = func(self._data)

        # If memory_efficient is True and the data type returned by func is
        # the same as the data type of the data itself,
        # take row-wise chunks of data, transform it, and put it back into
        # the original data, overwriting the original
        elif memory_efficient and _type_match:

            _n_chunks = int(np.ceil(self._adata.shape[0] / chunksize))

            for i in range(_n_chunks):
                _start = i * chunksize
                _stop = min(_start + chunksize, self._adata.shape[0])
                self._data[_start:_stop, :] = func(self._data[_start:_stop, :])

        # Apply function to the data
        # by making a new data object
        else:
            self._data = func(self._data)

    def apply(
        self,
        func,
        **kwargs
    ):
        """
        Apply a function that takes a 2d numpy array or sparse matrix to
        the underlying data. Replaces the data in place.
        Will pass any kwargs to the function.

        :param func: Function which takes an array input
        :type func: callable
        """

        self._adata.X = func(
            self._adata.X,
            **kwargs
        )

        return self

    def add(
        self,
        val,
        axis=None
    ):
        """
        Add a value to the matrix in-place
        :param val: Value to add
        :type val: numeric
        :param axis: Which axis to add to (0, 1, or None)
        :type axis: int, None
        """
        self._math_inplace_with_broadcasts(
            val,
            add=True,
            axis=axis
        )

    def subtract(
        self,
        val,
        axis=None
    ):
        """
        Subtract a value from the matrix in-place
        :param val: Value to subtract
        :type val: numeric
        :param axis: Which axis to subtract from (0, 1, or None)
        :type axis: int, None
        """
        self._math_inplace_with_broadcasts(
            val,
            subtract=True,
            axis=axis
        )

    def multiply(
        self,
        mult_val,
        axis=None
    ):
        """
        Multiply the matrix by a value in-place
        :param mult_val: Value to multiply
        :type mult_val: numeric
        :param axis: Which axis to multiply against (0, 1, or None)
        :type axis: int, None
        """
        self._math_inplace_with_broadcasts(
            mult_val,
            multiply=True,
            axis=axis
        )

    def divide(
        self,
        div_val,
        axis=None
    ):
        """
        Divide a value from the matrix in-place
        :param div_val: Value to divide
        :type div_val: numeric
        :param axis: Which axis to divide from (0, 1, or None)
        :type axis: int, None
        """

        self._math_inplace_with_broadcasts(
            div_val,
            divide=True,
            axis=axis
        )

    def _math_inplace_with_broadcasts(
        self,
        value,
        add=False,
        subtract=False,
        multiply=False,
        divide=False,
        axis=None
    ):
        """
        Do in-place math with broadcasting

        :param value: Value(s)
        :type value: numeric, np.ndarray
        :param add: Add, defaults to False
        :type add: bool, optional
        :param subtract: Subtract, defaults to False
        :type subtract: bool, optional
        :param multiply: Multiply, defaults to False
        :type multiply: bool, optional
        :param divide: Divide, defaults to False
        :type divide: bool, optional
        :param axis: Broadcast axis, defaults to None
        :type axis: int, optional
        """

        # Define in-place math function
        if add:
            def _mfunc(x):
                self._data += x
        elif subtract:
            def _mfunc(x):
                self._data -= x
        elif multiply:
            def _mfunc(x):
                self._data *= x
        elif divide:
            def _mfunc(x):
                self._data /= x
        else:
            raise ValueError(
                "Must set multiply=True or divide=True"
            )

        # Convert data to floats
        if self._is_integer:
            self.convert_to_float()

        # Modify in place if axis is None
        if axis is None:
            _mfunc(value)

        # Modify in place by being clever about repeating the division values
        # To align with the data object if it's a sparse matrix
        elif self.is_sparse and (axis == 0 or axis == 1):

            # Check the sparse type for validity
            _valid_type = sparse.isspmatrix_csr(self._adata.X) and axis == 1
            _valid_type |= sparse.isspmatrix_csc(self._adata.X) and axis == 0

            if not _valid_type:
                raise ValueError(
                    "axis = 1 is only valid for CSC matrices "
                    "and axis = 0 is only valid for CSR matrices; "
                    f"axis={axis} and {type(self._adata.X)} passed"
                )

            _invalid_dim = (
                not hasattr(value, "ndim") or
                value.ndim != 1 or
                self.shape[0 if axis else 1] != value.shape[0]
            )

            if _invalid_dim:
                raise ValueError(
                    "Value array is not aligned; "
                    f"{value.shape[0]} values provided against "
                    f"{self.shape[0 if axis else 1]} "
                    f"(axis={axis})"
                )

            _mfunc(
                np.repeat(
                    value,
                    self._adata.X.getnnz(axis=axis)
                )
            )

        # Divide in place by broadcasting
        elif axis == 0:
            _mfunc(value[None, :])

        elif axis == 1:
            _mfunc(value[:, None])

        else:
            raise ValueError("axis must be 0, 1 or None")

    def copy(self):

        new_data = InferelatorData(
            self.values.copy(),
            meta_data=self.meta_data.copy(),
            gene_data=self.gene_data.copy()
        )

        new_data._adata.var_names = self._adata.var_names.copy()
        new_data._adata.obs_names = self._adata.obs_names.copy()
        new_data._adata.uns = self._adata.uns.copy()

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

    def to_df(self):
        return self._adata.to_df()

    def replace_data(
            self,
            new_data,
            new_gene_names=None,
            new_gene_metadata=None
    ):

        if new_gene_metadata is None and new_gene_names is not None:
            new_gene_metadata = pd.DataFrame(index=new_gene_names)

        self._adata = AnnData(
            X=new_data,
            var=new_gene_metadata,
            obs=self._adata.obs
        )

        gc.collect()

    @staticmethod
    def _make_idx_str(df):

        if not pat.is_string_dtype(df.index.dtype):
            df.index = df.index.astype(str)

        if not pat.is_string_dtype(df.columns.dtype):
            df.columns = df.columns.astype(str)

    def _counts(self, axis=None):
        if self.is_sparse:
            return self._adata.X.sum(axis=axis).A.flatten()
        else:
            return self._adata.X.sum(axis=axis)

    def _means(self, axis=None):
        if self.is_sparse:
            return self._adata.X.mean(axis=axis).A.flatten()
        else:
            return self._adata.X.mean(axis=axis)

    def _vars(self, axis=None, ddof=1):
        if self.is_sparse:
            return StandardScaler(
                copy=False,
                with_mean=False
            ).fit(self._adata.X).var_
        else:
            return self._adata.X.var(axis=axis, ddof=ddof)

    def _stds(self, axis=None, ddof=1):
        if self.is_sparse:
            return np.sqrt(
                self._vars(axis=axis, ddof=ddof)
            )
        else:
            return self._adata.X.std(axis=axis, ddof=ddof)
