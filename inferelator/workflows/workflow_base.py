import datetime
import os
import warnings
import copy

import numpy as np
import pandas as pd

from inferelator.utils import (
    Debug,
    InferelatorDataLoader,
    DEFAULT_PANDAS_TSV_SETTINGS,
    slurm_envs,
    DotProduct
)

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.preprocessing import (
    ManagePriors,
    make_data_noisy
)
from inferelator.postprocessing import (
    ResultsProcessor,
    InferelatorResults as IR
)

SBATCH_VARS_FOR_WORKFLOW = ["output_dir", "input_dir"]

_TENX = "tenx"
_TSV = "tsv"
_H5AD = "h5ad"
_HDF5 = "hdf5"
_MTX = "mtx"

_VALID_FILE_TYPES = [_TSV, _H5AD, _HDF5, _MTX, _TENX]

class WorkflowBaseLoader(object):
    """
    WorkflowBaseLoader is the class to load raw data.
    It does no processing; it only takes data from files.
    """
    # Paths to the input and output locations
    input_dir = None
    output_dir = None

    # File names for each of the data files which
    # can be used in the inference workflow
    expression_matrix_file = None
    tf_names_file = None
    meta_data_file = None
    priors_file = None
    gold_standard_file = None

    # The expression file type
    _expression_loader = _TSV
    _h5_layer = None

    # Metadata handler
    metadata_handler = "branching"

    # Gene list & associated metadata
    gene_names_file = None
    gene_metadata_file = None
    gene_list_index = None

    # Loaded data structures [G: Genes, K: Predictors, N: Conditions]
    tf_names = None  # tf_names list [K]
    gene_names = None  # gene_names list [G]
    priors_data = None  # priors data dataframe [G x K]
    gold_standard = None  # gold standard dataframe [G x K]

    # Loaded experimental data
    data = None  # InferelatorData [N x G]

    # Calculated data structures
    design = None  # InferelatorData [N x K]
    response = None  # InferelatorData [N x G]

    # Flag to identify orientation of the expression matrix
    # (True for samples x genes & False for genes x samples)
    expression_matrix_columns_are_genes = True  # bool

    # Flag to extract metadata from specific columns of the
    # expression matrix instead of a separate file
    extract_metadata_from_expression_matrix = False  # bool
    expression_matrix_metadata = None  # str

    # Flag to indicate the inferelator should be run
    # without existing network data
    use_no_prior = False  # bool
    use_no_gold_standard = False  # bool

    # Settings that will be used by pd.read_table to
    # import data files
    _file_format_settings = None

    @property
    def _num_obs(self):
        """
        Get the number of observations N
        :return: Number of observations / samples
        :rtype: int
        """
        if self.response is not None:
            return self.response.num_obs
        elif self.data is not None:
            return self.data.num_obs
        else:
            return None

    @property
    def _num_tfs(self):
        """
        Get the number of regulators K
        :return: Number of regulators
        :rtype: int
        """
        if self.design is not None:
            return self.design.num_genes
        elif self.tf_names is not None:
            return len(self.tf_names)
        else:
            return None

    @property
    def _num_genes(self):
        """
        Get the number of genes G
        :return: Number of genes
        :rtype: int
        """
        if self.response is not None:
            return self.response.num_genes
        elif self.data is not None:
            return self.data.num_genes
        else:
            return None

    @property
    def _gene_names(self):
        if self.data is not None:
            return self.data.gene_names
        else:
            return None

    def __init__(self):
        if self._file_format_settings is None:
            self._file_format_settings = dict()

    def set_file_paths(
        self,
        input_dir=None,
        output_dir=None,
        expression_matrix_file=None,
        tf_names_file=None,
        meta_data_file=None,
        priors_file=None,
        gold_standard_file=None,
        gene_metadata_file=None,
        gene_names_file=None
    ):
        """
        Set the file paths necessary for the inferelator to run

        :param input_dir: A path containing the input files
        :type input_dir: str
        :param output_dir: A path to put the output files
        :type output_dir: str, optional
        :param expression_matrix_file: Path to the expression data
            If set here, this expression file will be assumed to be a TSV file.
            Use set_expression_file() for other file types
        :type expression_matrix_file: str
        :param meta_data_file: Path to the meta data TSV file
        :type meta_data_file: str, optional
        :param tf_names_file: Path to a list of regulator names to include
            in the model
        :type tf_names_file: str
        :param priors_file: Path to a prior data file TSV file
            [Genes x Regulators]
        :type priors_file: str
        :param gold_standard_file: Path to a gold standard data TSV
            file [Genes x Regulators]
        :type gold_standard_file: str
        :param gene_metadata_file: Path to a genes annotation file
        :type gene_metadata_file: str, optional
        :param gene_names_file: Path to a list of genes to include in
            the model (optional)
        :type gene_names_file: str, optional
        """

        self._set_with_warning("input_dir", input_dir)
        self._set_with_warning("output_dir", output_dir)
        self._set_file_name("expression_matrix_file", expression_matrix_file)
        self._set_file_name("tf_names_file", tf_names_file)
        self._set_file_name("meta_data_file", meta_data_file)
        self._set_file_name("priors_file", priors_file)
        self._set_file_name("gold_standard_file", gold_standard_file)
        self._set_file_name("gene_metadata_file", gene_metadata_file)
        self._set_file_name("gene_names_file", gene_names_file)

        if expression_matrix_file is not None:
            self._expression_loader = _TSV

            # Warn if it doesn't have 'tsv' in the filename
            if ".tsv" not in expression_matrix_file.lower():
                warnings.warn(
                    "`set_file_paths` assumes data is in a TSV. "
                    "Use `set_expression_file` for other formats"
                )

    def set_expression_file(
        self,
        tsv=None,
        hdf5=None,
        h5ad=None,
        tenx_path=None,
        mtx=None,
        mtx_barcode=None,
        mtx_feature=None,
        h5_layer=None
    ):
        """
        Set the type of expression data file.
        Current loaders include TSV, hdf5, h5ad (AnnData),
        and MTX sparse files.
        Only one of these loaders can be used; passing arguments
        for multiple loaders will raise a ValueError.

        :param tsv: A path to a TSV (or tsv.gz) file which can be
            loaded by pandas.read_csv()
        :type tsv: str, optional
        :param hdf5: A path to a hdf5 file which can be loaded by
            pandas.HDFStore
        :type hdf5: str, optional
        :param h5ad: A path to an AnnData hd5 file
        :type h5ad: str, optional
        :param tenx_path: A path to the folder containing the
            10x mtx, barcode, and feature files
        :type tenx_path: Path, optional
        :param mtx: A path to an mtx file
        :type mtx: str, optional
        :param mtx_barcode: A path to a list of observation
            names (i.e. barcodes, etc) for the mtx file
        :type mtx_barcode: str, optional
        :param mtx_feature: A path to a list of gene names
            for the mtx file
        :type mtx_feature: str, optional
        :param h5_layer: The layer (in an AnnData h5) or the
            store key (in an hdf5) file to use.
            Defaults to using the first key.
        :type h5_layer: str, optional
        """

        nones = [tsv is None, hdf5 is None, h5ad is None, tenx_path is None, mtx is None]

        if all(nones):
            Debug.vprint("No file provided", level=0)
        elif sum(nones) != (len(nones) - 1):
            raise ValueError(
                "Only one type of input expression file can be set"
            )

        if tsv is not None:
            self._set_file_name("expression_matrix_file", tsv)
            self._expression_loader = _TSV
        elif hdf5 is not None:
            self._set_file_name("expression_matrix_file", hdf5)
            self._expression_loader = _HDF5
            self._h5_layer = h5_layer
        elif h5ad is not None:
            self._set_file_name("expression_matrix_file", h5ad)
            self._expression_loader = _H5AD
            self._h5_layer = h5_layer
        elif mtx is not None:
            self._check_file_exists(mtx)
            self._check_file_exists(mtx_barcode)
            self._check_file_exists(mtx_feature)
            self.expression_matrix_file = (mtx, mtx_barcode, mtx_feature)
            self._expression_loader = _MTX
        elif tenx_path is not None:
            self.expression_matrix_file = tenx_path
            self._expression_loader = _TENX

    def set_file_properties(
        self,
        extract_metadata_from_expression_matrix=None,
        expression_matrix_metadata=None,
        expression_matrix_columns_are_genes=None,
        gene_list_index=None,
        metadata_handler=None
    ):
        """
        Set properties associated with the input data files

        :param extract_metadata_from_expression_matrix: A boolean flag that
            should be set to True if there is non-expression data in the
            expression matrix. If True, `expression_matrix_metadata` must
            be provided.
            Defaults to False.
        :type extract_metadata_from_expression_matrix: bool, optional
        :param expression_matrix_metadata: A list of columns which, if
            provided, will be removed from the expression matrix file and
            kept as metadata.
            Defaults to None.
        :type expression_matrix_metadata: list(str), optional
        :param expression_matrix_columns_are_genes: A boolean flag indicating
            the orientation of the expression matrix.
            False reads expression matrix [genes (rows) x samples (columns)].
            True reads expression matrix [samples (rows) x genes (columns)].
            Defaults to False.
        :type expression_matrix_columns_are_genes: bool, optional
        :param gene_list_index: The column name in the gene metadata file
            which corresponds to the gene labels in the expression and prior
            data files. Must be provided if `gene_metadata_file` was provided
            to `set_file_paths()`.
            Defaults to None.
        :type gene_list_index: str, optional
        :param metadata_handler: A string which identifies the specific
            metadata parsing method to use. Options include
            "branching" or "nonbranching". Defaults to "branching".
        :type metadata_handler: str
        """

        if extract_metadata_from_expression_matrix is not None:
            warnings.warn(
                "Set expression_matrix_metadata to extract columns",
                DeprecationWarning
            )

        self._set_without_warning(
            "expression_matrix_columns_are_genes",
            expression_matrix_columns_are_genes
        )

        self._set_with_warning(
            "expression_matrix_metadata",
            expression_matrix_metadata
        )

        self._set_with_warning(
            "gene_list_index",
            gene_list_index
        )

        self._set_with_warning(
            "metadata_handler",
            metadata_handler
        )

    def set_network_data_flags(
        self,
        use_no_prior=None,
        use_no_gold_standard=None
    ):
        """
        Set flags to skip using existing network data. Note that these
        flags will be ignored if network data is provided

        :param use_no_prior: Flag to indicate the inferelator should be run
            without existing prior data. Will create a mock prior with no
            information. Highly inadvisable. Defaults to False
        :type use_no_prior: bool
        :param use_no_gold_standard: Flag to indicate the inferelator should
            be run without existing gold standard data. Will create a mock
            gold standard with no information. Highly inadvisable.
            Defaults to False
        :type use_no_gold_standard: bool
        """

        if use_no_prior:
            warnings.warn(
                "Omitting prior network data is not recommended. "
                "Performance will be poor. Use at your own risk.",
                UserWarning
            )

        if use_no_gold_standard:
            warnings.warn(
                "Omitting gold standard is not recommended. "
                "Output performance metrics will be meaningless. "
                "Use at your own risk",
                UserWarning
            )

        self._set_without_warning("use_no_prior", use_no_prior)
        self._set_without_warning("use_no_gold_standard", use_no_gold_standard)

    def set_file_loading_arguments(
        self,
        file_name,
        **kwargs
    ):
        """
        Update the settings for a given file name. By default we assume all
        files, except expression data, can be read in as TSV files. Any
        arguments provided here will be passed to `pandas.read_csv()` for the
        file name provided.

        `set_file_loading_arguments('expression_matrix_file', sep=",")`
        will read the expression_matrix_file as a CSV.

        :param file_name: The name of the variable containing the file name
            (from `set_file_properties`)
        :type file_name: str
        :param kwargs: Arguments to be passed to `pandas.read_csv()`
        """

        # Check and see if file_name is actually an object attribute
        # holding a file name. Use that if so.
        file_name = self._get_file_name_from_attribute(file_name)
        if file_name is None:
            return

        self._file_format_settings[file_name].update(kwargs)
        self.print_file_loading_arguments(file_name)

    def print_file_loading_arguments(
        self,
        file_name
    ):
        """
        Print the settings that will be used to load a given file name.

        :param file_name: The name of the variable containing the file name
            (from `set_file_properties`)
        :type file_name: str
        """

        # Check and see if file_name is actually an object attribute
        # holding a file name. Use that if so.

        file_name = self._get_file_name_from_attribute(file_name)
        if file_name is None:
            return

        Debug.vprint(
            f"File {file_name} has the following settings:"
            "\n\t".join([
                str(k) + " = " + str(v)
                for k, v in self._file_format_settings[file_name].items()
            ]),
            level=0
        )

    def _get_file_name_from_attribute(
        self,
        file_reference
    ):
        """
        Check and see if a file name is an object attribute that holds
        a file namee

        :param file_name: str
        :return file_name: str
        """
        # Check and see if file_name is actually an object attribute
        # holding a file name. Use that if so.

        file_name = getattr(self, file_reference, None)

        if file_name is None:
            Debug.vprint(
                f"File {file_reference} is unknown",
                level=0
            )

        return file_name

    def _set_file_name(
        self,
        attr_name,
        file_name
    ):
        """
        Set a file name. Create a set of default loading parameters
        and store them in _file_format_settings.
        Also print a warning if the file doesn't exist

        :param attr_name: str
        :param file_name: str
        """

        if file_name is None:
            return

        if file_name not in self._file_format_settings:
            self._file_format_settings[file_name] = copy.copy(
                DEFAULT_PANDAS_TSV_SETTINGS
            )

        self._check_file_exists(file_name)
        self._set_with_warning(attr_name, file_name)

    def _check_file_exists(
        self,
        file_name
    ):
        """
        Print a warning if a file doesn't exist

        :param file_name: str
        """

        if file_name is None:
            return None
        elif not os.path.isfile(self.input_path(file_name)):
            warnings.warn(
                f"File {self.input_path(file_name)} does not exist",
                RuntimeWarning
            )

    def _set_without_warning(
        self,
        attr_name,
        value
    ):
        """
        Set an attribute in this workflow.

        :param attr_name: str
        :param value:
        """

        if value is None:
            return

        setattr(self, attr_name, value)

    def _set_with_warning(
        self,
        attr_name,
        value
    ):
        """
        Set an attribute name. Warn if it's already not None
        :param attr_name: str
        :param value:
        """

        if value is None:
            return

        current_value = getattr(self, attr_name)

        if current_value is not None and current_value != value:
            warnings.warn(
                f"Setting {attr_name}: replacing value {current_value} with "
                f"value {value}"
            )

        setattr(self, attr_name, value)

    def get_data(self):
        """
        Read data files in to data structures.
        """

        self.read_expression()
        self.read_tfs()
        self.read_priors()

        # Validate that necessary input settings exist
        self.validate_data()

    def append_to_path(
        self,
        var_name,
        to_append
    ):
        """
        Add a string to an existing path variable

        :param var_name: The name of the path variable (`input_dir` or `output_dir`)
        :type var_name: str
        :param to_append: The path to join to the end of the existing path variable
        :type to_append: str

        """

        path = getattr(self, var_name, None)

        if path is None:
            raise ValueError(
                f"Cannot append {to_append} to {var_name} (Which is None)"
            )

        setattr(self, var_name, os.path.join(path, to_append))

    def read_expression(
        self,
        expression_matrix_file=None,
        meta_data_file=None,
        gene_data_file=None
    ):
        """
        Read expression data into an InferelatorData object

        :param expression_matrix_file: Expression file, defaults to None
        :type expression_matrix_file: str, optional
        :param meta_data_file: Metadata file, defaults to None
        :type meta_data_file: str, optional
        :param gene_data_file: Gene metadata file, defaults to None
        :type gene_data_file: str, optional
        """

        if expression_matrix_file is None:
            expression_matrix_file = self.expression_matrix_file

        if meta_data_file is None:
            meta_data_file = self.meta_data_file

        if gene_data_file is None:
            gene_data_file = self.gene_metadata_file

        self.data = self.read_gene_data_file(
            expression_matrix_file,
            self._expression_loader,
            file_layer=self._h5_layer,
            meta_data_file=meta_data_file,
            gene_data_file=gene_data_file
        )

        self.data.name = "Expression"

    def read_gene_data_file(
        self,
        file_name,
        file_type,
        file_layer=None,
        meta_data_file=None,
        gene_data_file=None
    ):
        """
        Read file into an InferelatorData object
        """

        loader = InferelatorDataLoader(
            input_dir=self.input_dir,
            file_format_settings=self._file_format_settings
        )

        if file_type == _H5AD:
            return loader.load_data_h5ad(
                file_name,
                use_layer=file_layer,
                meta_data_file=meta_data_file,
                meta_data_handler=self.metadata_handler,
                gene_data_file=gene_data_file,
                gene_name_column=self.gene_list_index
            )

        elif file_type == _TSV:
            return loader.load_data_tsv(
                file_name,
                transpose_data=not self.expression_matrix_columns_are_genes,
                tsv_matrix_metadata=self.expression_matrix_metadata,
                meta_data_file=meta_data_file,
                meta_data_handler=self.metadata_handler,
                gene_data_file=gene_data_file,
                gene_name_column=self.gene_list_index
            )

        elif file_type == _MTX:
            return loader.load_data_mtx(
                file_name[0],
                mtx_feature=file_name[2],
                mtx_obs=file_name[1],
                meta_data_file=meta_data_file,
                meta_data_handler=self.metadata_handler,
                gene_data_file=gene_data_file,
                gene_name_column=self.gene_list_index
            )

        elif file_type == _TENX:
            return loader.load_data_tenx(
                file_name,
                meta_data_file=meta_data_file,
                meta_data_handler=self.metadata_handler,
                gene_data_file=gene_data_file,
                gene_name_column=self.gene_list_index
            )

        elif file_type == _HDF5:
            return loader.load_data_hdf5(
                file_name,
                transpose_data=not self.expression_matrix_columns_are_genes,
                use_layer=file_layer,
                meta_data_file=meta_data_file,
                meta_data_handler=self.metadata_handler,
                gene_data_file=gene_data_file,
                gene_name_column=self.gene_list_index
            )

        else:
            raise ValueError(
                f"file_type must be in {_VALID_FILE_TYPES}; "
                f"{file_type} provided"
            )

    def read_data_frame(self, file_name, **kwargs):

        if file_name is None:
            return None

        loader = InferelatorDataLoader(
            input_dir=self.input_dir,
            file_format_settings=self._file_format_settings
        )

        return loader.input_dataframe(
            file_name,
            **kwargs
        )

    def read_tfs(self, file=None):
        """
        Read tf names file into tf_names
        """

        # Load the class variable if no file is passed
        file = self.tf_names_file if file is None else file

        if file is not None:

            Debug.vprint(
                f"Loading TF feature names from file {file}",
                level=1
            )

            # Read in a dataframe with no header or index
            tfs = self.read_data_frame(file, header=None, index_col=None)

            # Cast the dataframe into a list
            assert tfs.shape[1] == 1
            self.tf_names = tfs.values.flatten().tolist()

    def read_genes(self, file=None):
        """
        Read gene names file into gene_names
        """

        # Load the class variable if no file is passed
        file = self.gene_names_file if file is None else file

        if file is not None:

            Debug.vprint(
                f"Loading Gene feature names from file {file}",
                level=1
            )

            # Read in a dataframe with no header or index
            genes = self.read_data_frame(file, header=None, index_col=None)

            # Cast the dataframe into a list
            assert genes.shape[1] == 1
            self.gene_names = genes.values.flatten().tolist()

        # Use the gene names in the data file if no restrictive list is provided
        if self.gene_names is None and self.data is not None:
            self.gene_names = self.data.gene_names.copy()

    def read_priors(self, priors_file=None, gold_standard_file=None):
        """
        Read in the priors and gold standard files
        """

        if priors_file is None:
            priors_file = self.priors_file

        if gold_standard_file is None:
            gold_standard_file = self.gold_standard_file

        if priors_file is not None:

            Debug.vprint(
                f"Loading prior data from file {priors_file}",
                level=1
            )

            self.priors_data = self.read_data_frame(priors_file)

            # Print debug info & check prior for duplicate
            # indices (which will raise errors later)
            self.loaded_file_info(
                "Priors data",
                self.priors_data
            )

            self._check_network_labels_unique(
                "Priors_data",
                priors_file,
                self.priors_data
            )

            # Add to the data object if loaded
            if self.data is not None:
                self.data.prior_data = self.priors_data.copy()

        if gold_standard_file is not None:

            Debug.vprint(
                f"Loading gold_standard data from file {gold_standard_file}",
                level=1
            )

            self.gold_standard = self.read_data_frame(gold_standard_file)

            # Print debug info & check gold standard for duplicate
            # indices (which will raise errors later)
            self.loaded_file_info(
                "Gold standard",
                self.gold_standard
            )

            self._check_network_labels_unique(
                "Gold standard",
                gold_standard_file,
                self.gold_standard
            )

    def validate_data(
        self,
        check_prior=True,
        check_gold_standard=True
    ):
        """
        Make sure that the data that's loaded is acceptable
        """

        if check_prior:

            # Create a null prior if the flag is set
            if self.use_no_prior and self.priors_data is not None:
                warnings.warn(
                    "use_no_prior flag will be ignored "
                    "because prior data exists"
                )

            elif self.use_no_prior:
                self.priors_data = self._create_null_prior(
                    self._gene_names,
                    self.tf_names
                )

                Debug.vprint(
                    f"A {self.priors_data.shape} null prior "
                    "has been created",
                    level=0
                )

        if check_gold_standard:

            # Create a null gold standard if the flag is set
            if self.use_no_gold_standard and self.gold_standard is not None:
                warnings.warn(
                    "use_no_gold_standard flag will be ignored "
                    "because gold standard data exists"
                )

            elif self.use_no_gold_standard:
                self.gold_standard = self._create_null_prior(
                    self._gene_names,
                    self.tf_names
                )

                Debug.vprint(
                    f"A {self.gold_standard.shape} null gold standard "
                    "has been constructed",
                    level=0
                )

            elif self.gold_standard is None:
                raise ValueError(
                    "No gold standard. Model scoring will be invalid. Set"
                    "worker.set_network_data_flags(use_no_gold_standard=True) "
                    "to explicitly continue."
                )

        if check_prior and check_gold_standard:
            # Validate that some network information exists
            # and has been loaded
            if self.priors_data is None and self.gold_standard is None:
                raise ValueError(
                    "No gold standard or priors have been provided"
                )

    def input_path(self, filename):
        """
        Join filename to input_dir

        :param filename: Path to some file that needs to be
            attached to the input path
        :type filename: str
        :return: File joined to input_dir instance variable
        :rtype: str
        """

        return InferelatorDataLoader.filename_path_join(
            self.input_dir,
            filename
        )

    def output_path(self, filename):
        """
        Join filename to output_dir

        :param filename: Path to some file that needs to be
            attached to the output path
        :type filename: str
        :return: File joined to output_dir instance variable
        :rtype: str
        """
        return InferelatorDataLoader.filename_path_join(
            self.output_dir,
            filename
        )

    def load_data_and_save_h5ad(self, output_file_name, to_sparse=False):
        """
        Load the workflow data and then save it to an h5ad file

        :param output_file_name: A path to the output file
        :type output_file_name: str
        :param to_sparse: Convert the data to sparse prior to saving it
        :type to_sparse: bool
        """

        self.read_expression()
        self.read_tfs()

        if to_sparse and not self.data.is_sparse:
            self.data.to_sparse()

        self.data.to_h5ad(self.output_path(output_file_name))

    @staticmethod
    def _create_null_prior(gene_names, tf_names):
        """
        Create a prior data matrix that is all 0s
        :param gene_names: Anything that can be used as an index
            for a dataframe
        :param tf_names: list, pd.Index
        :return priors: pd.DataFrame
        """

        if tf_names is None:
            raise ValueError(
                "Unable to generate a null prior without a TF list"
            )

        return pd.DataFrame(0, index=gene_names, columns=tf_names)

    @staticmethod
    def dataframe_split(data_frame, remove_columns):
        """
        Take a dataframe and extract specific columns. Return the dataframe,
        minus those columns, and a second dataframe which is only those
        columns.

        :param data_frame: pd.DataFrame
        :param remove_columns: list(str)
        :return data_frame, data_frame_two: pd.DataFrame, pd.DataFrame
        """

        data_frame_two = data_frame.loc[:, remove_columns].copy()
        data_frame = data_frame.drop(remove_columns, axis=1)

        return data_frame, data_frame_two

    @staticmethod
    def loaded_file_info(df_name, df):

        Debug.vprint(
            df_name + f" loaded {df.shape}",
            level=2
        )

        Debug.vprint(
            df_name + " index: " + str(df.index[0]) + " ...",
            level=2
        )

        Debug.vprint(
            df_name + " columns: " + str(df.columns[0]) + " ...",
            level=2
        )

    @staticmethod
    def _check_network_labels_unique(
        df_name,
        file_name,
        df,
        raise_on_duplicate=False
    ):

        _repeated = False

        # Check columns
        if not df.columns.is_unique:
            _repeated = df.columns[df.columns.duplicated()]
            _msg = f"{df_name} {file_name}: {len(_repeated)} "
            _msg += f"TFs are duplicated ({' '.join(_repeated)})"
        else:
            _msg = ""

        # Then check rows
        if not df.index.is_unique:
            _repeated = df.index[df.index.duplicated()]
            _msg += f"{df_name} {file_name}: {len(_repeated)} "
            _msg += f"genes are duplicated ({' '.join(_repeated)})"

        # If they weren't unique, raise a ValueError
        # Or just print depending on the flag
        if _repeated and raise_on_duplicate:
            raise ValueError(_msg)
        elif _repeated:
            Debug.vprint(_msg, level=0)


class WorkflowBase(WorkflowBaseLoader):
    """
    WorkflowBase handles crossvalidation, shuffling, and
    validating priors and gold standards
    """
    # Flags to control splitting priors into a prior/gold-standard set
    split_gold_standard_for_crossvalidation = False
    cv_split_ratio = None
    cv_split_axis = 0

    # Flags to control shuffle and noise baselines
    shuffle_prior_axis = None
    make_data_noise = None
    add_prior_noise = None
    _data_is_noise = False

    # The random seed for sampling, etc
    random_seed = 42

    # The number of inference bootstraps to run
    num_bootstraps = 2

    # Use the Intel MKL libraries for matrix multiplication
    use_mkl = None

    # Use numba for JIT
    use_numba = True

    # Multiprocessing controller
    initialize_mp = True
    multiprocessing_controller = None

    # Prior manager
    prior_manager = ManagePriors

    # Result processing & model metrics
    _result_processor_driver = ResultsProcessor
    gold_standard_filter_method = "keep_all_gold_standard"
    metric = "combined"

    # Output results in an InferelatorResults object
    results = None

    def __init__(self):
        super(WorkflowBase, self).__init__()
        # Get environment variables
        self.get_environmentals()

    def set_crossvalidation_parameters(
        self,
        split_gold_standard_for_crossvalidation=None,
        cv_split_ratio=None,
        cv_split_axis=None
    ):
        """
        Set parameters for crossvalidation.

        :param split_gold_standard_for_crossvalidation: Boolean flag indicating
            if the gold standard should be split. Must be set to True for other
            crossvalidation settings to have an effect. Defaults to False.
        :type split_gold_standard_for_crossvalidation: bool
        :param cv_split_ratio: The proportion of the gold standard which should
            be retained for scoring. The rest will be used to train the model.
            Must be set betweeen 0 and 1.
        :type cv_split_ratio: float
        :param cv_split_axis: How to split the gold standard.

            If 0, split genes; this will take all the data for certain genes
            and keep it in the gold standard. These genes will be removed from
            the prior.

            If 1, split regulators; this will take all the data for certain
            regulatorsnand keep it in the gold standard. These regulators will
            be removed from the prior. Splitting regulators is inadvisable.

            If None, the prior will be replaced with a downsampled gold
            standard.

            Setting this to 0 is generally the best choice. Defaults to None.

        :type cv_split_axis: int, None

        """

        self._set_without_warning(
            "split_gold_standard_for_crossvalidation",
            split_gold_standard_for_crossvalidation
        )
        self._set_with_warning(
            "cv_split_ratio",
            cv_split_ratio
        )
        self._set_with_warning(
            "cv_split_axis",
            cv_split_axis
        )

        if cv_split_axis is not None or cv_split_ratio is not None:

            if not self.split_gold_standard_for_crossvalidation:
                warnings.warn(
                    "The split_gold_standard_for_crossvalidation flag is not set. "
                    "Other options may be ignored"
                )

    def set_shuffle_parameters(
        self,
        shuffle_prior_axis=None,
        make_data_noise=None,
        add_prior_noise=None
    ):
        """
        Set parameters for shuffling labels on a prior axis. This is useful
        to establish a baseline.

        :param shuffle_prior_axis: The axis for shuffling prior labels.
            0 shuffles gene labels.
            1 shuffles regulator labels.
            None means labels will not be shuffled.
            Defaults to None.
        :type shuffle_prior_axis: int, None
        :param make_data_noise: Replace loaded data with simulated data
            that is entirely random. This retains type; integer data remains
            integer, float remains float. Gene distributions should be
            centered around the mean of gene expression in the original data,
            but is otherwise random.
        :type make_data_noise: bool, None
        :param add_prior_noise: Add random edges to the prior data.
            This is a numeric value between 0 and 1 such that
            0 adds no edges,
            1 sets every edge in the prior to True,
            0.1 sets 10% of the edges in the prior to True,
            and so on.
            Note that this will binarize the prior if it is not already binary.
        :type add_prior_noise: numeric, None
        """
        self._set_with_warning("shuffle_prior_axis", shuffle_prior_axis)
        self._set_with_warning("make_data_noise", make_data_noise)
        self._set_with_warning("add_prior_noise", add_prior_noise)

    def set_postprocessing_parameters(
        self,
        gold_standard_filter_method=None,
        metric=None
    ):
        """
        Set parameters for the postprocessing engine

        :param gold_standard_filter_method: A flag that determines if the
            old standard should be shrunk to the size of the produced model.
            "overlap" will only score on overlap between the gold standard
            and the inferred gene regulatory network.
            "keep_all_gold_standard" will score on the entire gold standard.
            Defaults to "keep_all_gold_standard".
        :type gold_standard_filter_method: str
        :param metric: The model metric to use for scoring. Supports
            "precision-recall", "mcc", "f1", and "combined"
            Defaults to "combined".
        :type metric: str
        """

        self._set_with_warning(
            "gold_standard_filter_method",
            gold_standard_filter_method
        )
        self._set_with_warning(
            "metric",
            metric
        )

    @staticmethod
    def set_output_file_names(
        network_file_name="",
        confidence_file_name="",
        nonzero_coefficient_file_name="",
        pdf_curve_file_name="",
        curve_data_file_name=""
    ):
        """
        Set output file names. File names that end in '.gz' will be gzipped.

        :param network_file_name: Long-format network TSV file with
            TF->Gene edge information.
            Default is "network.tsv".
        :type network_file_name: str
        :param confidence_file_name: Genes x TFs TSV with confidence
            scores for each edge.
            Default is "combined_confidences.tsv"
        :type confidence_file_name: str
        :param nonzero_coefficient_file_name: Genes x TFs TSV with
            the non-zero model coefficients for each
            edge. Default is "model_coefficients.tsv.gz"
        :type nonzero_coefficient_file_name: str
        :param pdf_curve_file_name: PDF file with plotted curve(s).
            Default is "combined_metrics.pdf".
        :type pdf_curve_file_name: str
        :param curve_data_file_name: TSV file with the data used to plot
            curves. Default is None (this file is not produced).
        :type curve_data_file_name: str
        """

        if network_file_name != "":
            IR.network_file_name = network_file_name
        if confidence_file_name != "":
            IR.confidence_file_name = confidence_file_name
        if nonzero_coefficient_file_name != "":
            IR.threshold_file_name = nonzero_coefficient_file_name
        if pdf_curve_file_name != "":
            IR.curve_file_name = pdf_curve_file_name
        if curve_data_file_name != "":
            IR.curve_data_file_name = curve_data_file_name

    def set_run_parameters(
        self,
        num_bootstraps=None,
        random_seed=None,
        use_mkl=None,
        use_numba=None
    ):
        """
        Set parameters used during runtime

        :param num_bootstraps: The number of bootstraps to run.
            Defaults to 2.
        :type num_bootstraps: int
        :param random_seed: The random number seed to use.
            Defaults to 42.
        :type random_seed: int
        :param use_mkl: A flag to indicate if the intel MKL library
            should be used for matrix multiplication, defaults to False
        :type use_mkl: bool
        :param use_numba: A flag to indicate if numba should be used
            to accelerate the calculations. Requires numba to be installed
            if set. Currently only accelerates AMuSR regression, defaults
            to True
        :type use_numba: bool
        """

        self._set_without_warning("num_bootstraps", num_bootstraps)
        self._set_without_warning("random_seed", random_seed)
        self._set_without_warning("use_mkl", use_mkl)
        self._set_without_warning("use_numba", use_numba)

    def initialize_multiprocessing(self):
        """
        Register the multiprocessing controller if set and run .connect()
        """
        if self.multiprocessing_controller is not None:
            MPControl.set_multiprocess_engine(self.multiprocessing_controller)

        MPControl.connect()

    def get_environmentals(self):
        """
        Load environmental variables into class variables
        """
        for k, v in slurm_envs(SBATCH_VARS_FOR_WORKFLOW).items():
            setattr(self, k, v)

    def startup(self):
        """
        Startup by preprocessing all data into a ready format for regression.
        """

        DotProduct.set_mkl(self.use_mkl)

        if self.initialize_mp:
            self.initialize_multiprocessing()

        self.startup_run()
        self.startup_finish()

    def startup_run(self):
        """
        Execute any data preprocessing necessary before regression.
        Startup_run is mostly for reading in data
        """
        raise NotImplementedError  # implement in subclass

    def startup_finish(self):
        """
        Execute any data preprocessing necessary before regression.
        Startup_finish is mostly for preprocessing data
        prior to regression
        """
        raise NotImplementedError  # implement in subclass

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        raise NotImplementedError  # implement in subclass

    def process_priors_and_gold_standard(self):
        """
        Run split, shuffle, etc called for by the workflow flags
        This also filters the expression matrix to the list of genes to model
        """

        pm = self.prior_manager

        # Split gold standard for cross-validation
        if self.split_gold_standard_for_crossvalidation:
            self.priors_data, self.gold_standard = pm.cross_validate_gold_standard(
                self.priors_data,
                self.gold_standard,
                self.cv_split_axis,
                self.cv_split_ratio,
                self.random_seed
            )

        # Filter priors to a list of regulators
        if self.tf_names is not None:
            self.priors_data = pm.filter_to_tf_names_list(
                self.priors_data,
                self.tf_names
            )

        elif self.tf_names is None and self.priors_data is not None:
            self.tf_names = self.priors_data.columns.tolist()

        elif self.tf_names is None:
            raise ValueError(
                "Either a priors_data or a tf_names file must be provided "
                "to identify regulators."
            )

        # Filter priors and expression to a list of genes
        self.filter_to_gene_list()

        # Shuffle prior labels
        if self.shuffle_prior_axis is not None:
            self.priors_data = pm.shuffle_priors(
                self.priors_data,
                self.shuffle_prior_axis,
                self.random_seed
            )

        # Check for duplicates or whatever
        self.priors_data, self.gold_standard = pm.validate_priors_gold_standard(
            self.priors_data,
            self.gold_standard
        )

        if self.add_prior_noise is not None:
            self.priors_data = pm.add_prior_noise(
                self.priors_data,
                self.add_prior_noise,
                self.random_seed
            )

    def filter_to_gene_list(self):
        """
        Filter the priors and expression matrix to just genes in gene_metadata
        """
        self.data.trim_genes(trim_gene_list=self.gene_names)
        self.priors_data = self.prior_manager.filter_priors_to_genes(
            self.priors_data,
            self.data.gene_names
        )

    def align_priors_and_expression(self):
        """
        Align prior to the expression matrix
        """
        self.priors_data = self.prior_manager.align_priors_to_expression(
            self.priors_data,
            self.data.gene_names
        )
        self.data_white_noise()

    def data_white_noise(self):
        """
        Replace data with white noise data
        """

        if self.make_data_noise and not self._data_is_noise:
            make_data_noisy(self.data, random_seed=self.random_seed)
            self._data_is_noise = True

    def get_bootstraps(self):
        """
        Generate sequence of bootstrap parameter objects for run.
        """
        return np.random.RandomState(seed=self.random_seed).choice(
            range(self._num_obs),
            size=(self.num_bootstraps, self._num_obs)
        ).tolist()

    def emit_results(
        self,
        betas,
        rescaled_betas,
        gold_standard,
        priors,
        full_model=None,
        full_exp_var=None
    ):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass

    def create_output_dir(self):
        """
        Set a default output_dir if nothing is set.
        Create the path if it doesn't exist.
        """

        if self.output_dir is None:
            self.output_dir = InferelatorDataLoader.make_path_safe(
                os.path.join(
                    self.input_dir,
                    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                )
            )

        else:
            self.output_dir = InferelatorDataLoader.make_path_safe(
                self.output_dir
            )

        # Create the output directory
        try:
            os.makedirs(self.output_dir)
        except FileExistsError:
            pass

    def create_task(self, **kwargs):
        """
        Create a task data object
        """
        raise NotImplementedError(
            "This workflow does not support multiple tasks"
        )
