import pandas as pd
import os
import copy as cp

from inferelator.utils.data import InferelatorData
from inferelator.utils.debug import Debug
from inferelator.preprocessing.metadata_parser import MetadataHandler

DEFAULT_PANDAS_TSV_SETTINGS = dict(sep="\t", index_col=0, header=0)
DEFAULT_METADATA = "branching"


class InferelatorDataLoader(object):
    input_dir = None
    _file_format_settings = None

    def __init__(self, input_dir, file_format_settings=None):
        self.input_dir = input_dir
        self._file_format_settings = file_format_settings

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

        # Load metadata
        meta_data_handler = MetadataHandler.get_handler(meta_data_handler)
        if meta_data_file is not None:
            Debug.vprint("Loading metadata file {file}".format(file=meta_data_file), level=0)
            meta_data = meta_data_handler.check_loaded_meta_data(self.input_dataframe(meta_data_file, index_col=None))
        elif slice_meta_data is None:
            Debug.vprint("No metadata provided. Creating a generic metadata", level=0)
            sample_labels = data.columns if transpose_expression_data else data.index
            meta_data = meta_data_handler.create_default_meta_data(sample_labels)
        else:
            meta_data = None

        meta_data = pd.concat((meta_data, slice_meta_data), axis=1)

        # Load gene metadata
        if gene_data_file is None and gene_name_column is None:
            gene_metadata = None
        elif gene_data_file is None or gene_name_column is None:
            raise ValueError("Gene_metadata_file and gene_list_index must both be set if either is")
        else:
            Debug.vprint("Loading gene metadata from file {file}".format(file=gene_data_file), level=0)
            gene_metadata = self.input_dataframe(gene_data_file, index_col=None)

            # Validate that the gene_metadata can be properly read, if loaded
            if gene_name_column in gene_metadata:
                gene_metadata.index = gene_metadata[gene_name_column]
            else:
                msg = "Column {c} not found in gene data file [{h}]".format(c=gene_name_column,
                                                                            h=" ".join(gene_metadata.columns))
                raise ValueError(msg)

        # Pack all data structures into an InferelatorData object
        data = InferelatorData(data,
                               transpose_expression=transpose_expression_data,
                               meta_data=meta_data,
                               gene_data=gene_metadata)

        nnf, non_finite_genes = data.non_finite
        if nnf > 0:
            Debug.vprint("{n} genes with non-finite expression ({g})".format(n=nnf, g=" ".join(non_finite_genes)))

        Debug.vprint("Expression data loaded: {dt}".format(dt=str(data)))
        return data

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
