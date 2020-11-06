import pandas as pd
import pandas.api.types as pat
import numpy as np
import os
import copy as cp
import anndata

from inferelator.utils.data import InferelatorData
from inferelator.utils.debug import Debug
from inferelator.preprocessing.metadata_parser import MetadataHandler

DEFAULT_PANDAS_TSV_SETTINGS = dict(sep="\t", index_col=0, header=0)
DEFAULT_METADATA = "branching"

_TENX_MTX = ("matrix.mtx.gz", "matrix.mtx")
_TENX_BARCODES = ("barcodes.tsv.gz", "barcodes.tsv")
_TENX_FEATURES = ("features.tsv.gz", "genes.tsv")


class InferelatorDataLoader(object):
    input_dir = None
    _file_format_settings = None

    def __init__(self, input_dir, file_format_settings=None):
        self.input_dir = input_dir
        self._file_format_settings = file_format_settings

    def load_data_h5ad(self, h5ad_file, meta_data_file=None, meta_data_handler=DEFAULT_METADATA, gene_data_file=None,
                       gene_name_column=None, use_layer=None):

        data = anndata.read_h5ad(self.input_path(h5ad_file))

        if meta_data_file is None and data.obs.shape[1] > 0:
            meta_data = None
        else:
            meta_data = self.load_metadata_tsv(meta_data_file, data.obs_names, meta_data_handler=meta_data_handler)

        gene_metadata = self.load_gene_metadata_tsv(gene_data_file, gene_name_column)

        if use_layer is not None and use_layer not in data.layers:
            msg = "Layer {lay} is not in {f}".format(lay=use_layer, f=h5ad_file)
            raise ValueError(msg)

        # Build an InferelatorData object from a layer
        elif use_layer is not None:
            data = InferelatorData(data.layers[use_layer],
                                   gene_names=data.var_names,
                                   sample_names=data.obs_names,
                                   meta_data=pd.concat((data.obs, meta_data), axis=1),
                                   gene_data=pd.concat((data.var, gene_metadata), axis=1))

        # Build an InferelatorData object from everything
        else:
            data = InferelatorData(data,
                                   meta_data=meta_data,
                                   gene_data=gene_metadata)

        # Make sure bytestrings are decoded
        _safe_dataframe_decoder(data.gene_data)
        _safe_dataframe_decoder(data.meta_data)

        self._check_loaded_data(data, filename=h5ad_file)
        return data

    def load_data_mtx(self, mtx_file, mtx_obs=None, mtx_feature=None, meta_data_file=None,
                      meta_data_handler=DEFAULT_METADATA, gene_data_file=None, gene_name_column=None):

        data = anndata.read_mtx(self.input_path(mtx_file))
        row_names = self._load_list_from_file(self.input_path(mtx_obs)) if mtx_obs is not None else None
        col_names = self._load_list_from_file(self.input_path(mtx_feature)) if mtx_feature is not None else None

        meta_data = self.load_metadata_tsv(meta_data_file, data.obs_names, meta_data_handler=meta_data_handler)
        gene_metadata = self.load_gene_metadata_tsv(gene_data_file, gene_name_column)

        data = InferelatorData(data,
                               meta_data=meta_data,
                               gene_data=gene_metadata,
                               sample_names=row_names,
                               gene_names=col_names)

        return data

    def load_data_hdf5(self, hdf5_file, use_layer=None, meta_data_file=None, meta_data_handler=DEFAULT_METADATA,
                       gene_data_file=None, gene_name_column=None, transpose_expression_data=False):

        data = pd.HDFStore(self.input_path(hdf5_file), mode='r')
        data = data[data.keys()[0]] if use_layer is None else data[use_layer]

        meta_data = self.load_metadata_tsv(meta_data_file, data.index, meta_data_handler=meta_data_handler)
        gene_metadata = self.load_gene_metadata_tsv(gene_data_file, gene_name_column)

        data = data.transpose() if transpose_expression_data else data
        data = InferelatorData(data,
                               meta_data=meta_data,
                               gene_data=gene_metadata)

        # Make sure bytestrings are decoded
        _safe_dataframe_decoder(data.gene_data)
        _safe_dataframe_decoder(data.meta_data)

        return data

    def load_data_tenx(self, tenx_path, meta_data_file=None, meta_data_handler=DEFAULT_METADATA, gene_data_file=None,
                       gene_name_column=None):

        mtx_file, mtx_obs, mtx_feature = None, None, None

        for datafile in _TENX_MTX:
            if self._file_exists(self.filename_path_join(tenx_path, datafile)):
                mtx_file = self.filename_path_join(tenx_path, datafile)

        for datafile in _TENX_BARCODES:
            if self._file_exists(self.filename_path_join(tenx_path, datafile)):
                mtx_obs = self.filename_path_join(tenx_path, datafile)

        for datafile in _TENX_FEATURES:
            if self._file_exists(self.filename_path_join(tenx_path, datafile)):
                mtx_feature = self.filename_path_join(tenx_path, datafile)

        if mtx_file is None:
            msg = "Cannot find 10x files ({f}) in path ({p})".format(f=" or ".join(_TENX_MTX), p=tenx_path)
            raise FileNotFoundError(msg)
        else:
            return self.load_data_mtx(mtx_file, mtx_obs=mtx_obs, mtx_feature=mtx_feature, meta_data_file=meta_data_file,
                                      meta_data_handler=meta_data_handler, gene_data_file=gene_data_file,
                                      gene_name_column=gene_name_column)

    def load_data_tsv(self, expression_matrix_file, transpose_expression_data=False, meta_data_file=None,
                      meta_data_handler=DEFAULT_METADATA, expression_matrix_metadata=None, gene_data_file=None,
                      gene_name_column=None):

        Debug.vprint("Loading expression data file {file}".format(file=expression_matrix_file), level=0)

        # Load expression data
        data = self.input_dataframe(expression_matrix_file)
        if expression_matrix_metadata is not None:
            meta_cols = data.columns.intersection(expression_matrix_metadata)
            slice_meta_data = data.loc[:, meta_cols].copy()
            data = data.drop(meta_cols, axis=1)
        else:
            slice_meta_data = None

        if meta_data_file is None and slice_meta_data is not None:
            meta_data = None
        else:
            sample_labels = data.columns if transpose_expression_data else data.index
            meta_data = self.load_metadata_tsv(meta_data_file, sample_labels, meta_data_handler=meta_data_handler)

        meta_data = pd.concat((meta_data, slice_meta_data), axis=1)

        gene_metadata = self.load_gene_metadata_tsv(gene_data_file, gene_name_column)

        # Pack all data structures into an InferelatorData object
        data = InferelatorData(data,
                               transpose_expression=transpose_expression_data,
                               meta_data=meta_data,
                               gene_data=gene_metadata)

        self._check_loaded_data(data, filename=expression_matrix_file)
        return data

    def load_metadata_tsv(self, meta_data_file, sample_labels, meta_data_handler=None):
        # Load metadata
        meta_data_handler = MetadataHandler.get_handler(meta_data_handler)
        if meta_data_file is not None:
            Debug.vprint("Loading metadata file {file}".format(file=meta_data_file), level=0)
            meta_data = meta_data_handler.check_loaded_meta_data(self.input_dataframe(meta_data_file, index_col=None))
        else:
            Debug.vprint("No metadata provided. Creating a generic metadata", level=0)
            meta_data = meta_data_handler.create_default_meta_data(sample_labels)

        return meta_data

    def load_gene_metadata_tsv(self, gene_data_file, gene_name_column):
        # Load gene metadata
        if gene_data_file is None and gene_name_column is None:
            return None
        elif gene_data_file is None or gene_name_column is None:
            raise ValueError("Gene_metadata_file and gene_list_index must both be set if either is")

        Debug.vprint("Loading gene metadata from file {file}".format(file=gene_data_file), level=0)
        gene_metadata = self.input_dataframe(gene_data_file, index_col=None)

        # Validate that the gene_metadata can be properly read, if loaded
        if gene_name_column in gene_metadata:
            gene_metadata.index = gene_metadata[gene_name_column]
        else:
            msg = "Column {c} not found in gene data file [{h}]".format(c=gene_name_column,
                                                                        h=" ".join(gene_metadata.columns))
            raise ValueError(msg)

        return gene_metadata

    def input_dataframe(self, filename, **kwargs):
        """
        Read a file in as a pandas dataframe
        """
        Debug.vprint("Loading data file: {a}".format(a=self.input_path(filename)), level=2)

        # Use any kwargs for this function and any file settings from default
        if self._file_format_settings is not None and filename in self._file_format_settings:
            file_settings = self._file_format_settings[filename]
        else:
            file_settings = cp.copy(DEFAULT_PANDAS_TSV_SETTINGS)

        file_settings.update(kwargs)

        # Load a dataframe
        return pd.read_csv(self.input_path(filename), **file_settings)

    def input_path(self, filename):
        """
        Join filename to input_dir

        :param filename: Path to some file that needs to be attached to the input path
        :type filename: str
        :return: File joined to input_dir instance variable
        :rtype: str
        """

        return self.filename_path_join(self.input_dir, filename)

    @staticmethod
    def _file_exists(filename):
        return filename is not None and os.path.isfile(filename)

    @staticmethod
    def _load_list_from_file(filename):
        return pd.read_csv(filename, sep="\t", header=None)[0].tolist() if filename is not None else None

    @staticmethod
    def _check_loaded_data(data, filename=None):
        msg = "Loaded {f}:\n".format(f=filename) if filename is not None else ""

        nnf, non_finite_genes = data.non_finite
        if nnf > 0:
            msg += "\t{n} genes with non-finite expression ({g})\n".format(n=nnf, g=" ".join(non_finite_genes))

        msg += "Data loaded: {dt}".format(dt=str(data))
        Debug.vprint(msg, level=0)

    @staticmethod
    def filename_path_join(path, filename):
        """
        Join filename to path
        """

        # Raise an error if filename is None
        if filename is None:
            raise ValueError("Cannot create a path to a filename set as None")

        # Return an absolute path unchanged
        elif os.path.isabs(filename):
            return filename

        # If path is set, join filename to it and return that
        elif path is not None:
            return InferelatorDataLoader.make_path_safe(os.path.join(path, filename))

        # If path is not set, convert the filename to absolute and return it
        else:
            return InferelatorDataLoader.make_path_safe(filename)

    @staticmethod
    def make_path_safe(path):
        """
        Expand relative paths to absolute paths. Pass None through.
        :param path: str
        :return: str
        """
        if path is not None:
            return os.path.abspath(os.path.expanduser(path))
        else:
            return None


def _safe_dataframe_decoder(data_frame, encoding='utf-8'):
    """
    Decode dataframe bytestrings

    :param data_frame: pd.DataFrame
    """

    if _is_dtype_object(data_frame.index.dtype):
        data_frame.index = _decode_series(data_frame.index, encoding=encoding)

    if _is_dtype_object(data_frame.columns.dtype):
        data_frame.columns = _decode_series(data_frame.columns, encoding=encoding)

    for col in data_frame.columns:
        if _is_dtype_object(data_frame[col].dtype):
            data_frame[col] = _decode_series(data_frame[col], encoding=encoding)


def _is_dtype_object(dtype):
    if pat.is_object_dtype(dtype):
        return True
    elif pat.is_categorical_dtype(dtype):
        return pat.is_object_dtype(dtype.categories.dtype)
    else:
        return False


def _decode_series(series, encoding):
    """
    Decode and return a series or index object from pandas

    :param series: pd.Series, pd.Index
    :param encoding: str
    :return: pd.Series, pd.Index
    """

    if pat.is_categorical_dtype(series):
        series.cat.categories = _decode_series(series.dtype.categories, encoding=encoding)
        return series

    _new_series = series.str.decode(encoding).values
    _no_decode = pd.isna(_new_series)

    if np.all(_no_decode):
        return series
    
    _new_series[_no_decode] = series.values[_no_decode]

    try:
        new_series = pd.Series(_new_series, index=series.index)
    except AttributeError:
        new_series = pd.Index(_new_series)

    return new_series
