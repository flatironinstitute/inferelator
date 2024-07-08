"""
Test base workflow stepwise.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sps
import pandas.testing as pdt
import numpy.testing as npt

from inferelator import workflow
from inferelator.regression.base_regression import _RegressionWorkflowMixin
from inferelator.distributed.inferelator_mp import MPControl
from inferelator.preprocessing.metadata_parser import MetadataParserBranching
from inferelator.utils import todense

my_dir = os.path.dirname(__file__)

DEFAULT_EXPRESSION_FILE = "expression.tsv"
DEFAULT_TFNAMES_FILE = "tf_names.tsv"
DEFAULT_METADATA_FILE = "meta_data.tsv"
DEFAULT_PRIORS_FILE = "gold_standard.tsv"
DEFAULT_GOLDSTANDARD_FILE = "gold_standard.tsv"


class TestWorkflowSetParameters(unittest.TestCase):

    def setUp(self):
        self.workflow = workflow.WorkflowBase()

    def test_set_file_names(self):
        self.assertIsNone(self.workflow.expression_matrix_file)
        self.assertIsNone(self.workflow.tf_names_file)
        self.assertIsNone(self.workflow.meta_data_file)
        self.assertIsNone(self.workflow.priors_file)
        self.assertIsNone(self.workflow.gold_standard_file)
        self.assertIsNone(self.workflow.input_dir)
        self.assertIsNone(self.workflow.output_dir)

        with self.assertWarns(Warning):
            self.workflow.set_file_paths(expression_matrix_file="A",
                                         tf_names_file="B",
                                         meta_data_file="C",
                                         priors_file="D",
                                         gold_standard_file="E",
                                         gene_metadata_file="F",
                                         input_dir="G",
                                         output_dir="H")

        self.assertListEqual([self.workflow.expression_matrix_file,
                              self.workflow.tf_names_file,
                              self.workflow.meta_data_file,
                              self.workflow.priors_file,
                              self.workflow.gold_standard_file,
                              self.workflow.gene_metadata_file,
                              self.workflow.input_dir,
                              self.workflow.output_dir],
                             ["A", "B", "C", "D", "E", "F", "G", "H"])

        with self.assertWarns(Warning):
            self.workflow.set_file_paths(expression_matrix_file="K")
            self.assertEqual(self.workflow.expression_matrix_file, "K")

    def test_set_file_properties(self):
        self.assertTrue(self.workflow.expression_matrix_columns_are_genes)
        self.assertIsNone(self.workflow.expression_matrix_metadata)
        self.assertIsNone(self.workflow.gene_list_index)

        self.workflow.set_file_properties(expression_matrix_columns_are_genes=True,
                                          expression_matrix_metadata=["A"],
                                          gene_list_index=["B"])

        self.assertTrue(self.workflow.expression_matrix_columns_are_genes)
        self.assertListEqual(self.workflow.expression_matrix_metadata, ["A"])
        self.assertListEqual(self.workflow.gene_list_index, ["B"])

        with self.assertWarns(Warning):
            self.workflow.set_file_properties(expression_matrix_metadata=["K"])
            self.assertListEqual(self.workflow.expression_matrix_metadata, ["K"])

        with self.assertWarns(DeprecationWarning):
            self.workflow.set_file_properties(extract_metadata_from_expression_matrix=True)

    def test_set_network_flags(self):
        self.assertFalse(self.workflow.use_no_prior)
        self.assertFalse(self.workflow.use_no_gold_standard)

        with self.assertRaises(AssertionError):
            with self.assertWarns(UserWarning):
                self.workflow.set_network_data_flags()

        with self.assertWarns(UserWarning):
            self.workflow.set_network_data_flags(use_no_gold_standard=True)

        with self.assertWarns(UserWarning):
            self.workflow.set_network_data_flags(use_no_prior=True)

        self.assertTrue(self.workflow.use_no_prior)
        self.assertTrue(self.workflow.use_no_gold_standard)

    def test_set_cv_params(self):
        self.assertIsNone(self.workflow.cv_split_ratio)
        self.assertFalse(self.workflow.split_gold_standard_for_crossvalidation)

        with self.assertWarns(Warning):
            self.workflow.set_crossvalidation_parameters(cv_split_ratio=0.2)

        self.assertEqual(self.workflow.cv_split_ratio, 0.2)
        self.workflow.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True)
        self.assertTrue(self.workflow.split_gold_standard_for_crossvalidation)

    def test_set_run_params(self):
        self.workflow.set_run_parameters(num_bootstraps=12345678, random_seed=87654321)
        self.assertEqual(self.workflow.num_bootstraps, 12345678)
        self.assertEqual(self.workflow.random_seed, 87654321)

    def test_set_postprocessing_params(self):
        with self.assertWarns(Warning):
            self.workflow.set_postprocessing_parameters(gold_standard_filter_method="red", metric="blue")
        self.assertListEqual([self.workflow.gold_standard_filter_method, self.workflow.metric], ["red", "blue"])


class TestWorkflowLoadData(unittest.TestCase):

    def setUp(self):
        self.workflow = workflow.WorkflowBase()
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.workflow.expression_matrix_file = DEFAULT_EXPRESSION_FILE
        self.workflow.tf_names_file = DEFAULT_TFNAMES_FILE
        self.workflow.meta_data_file = DEFAULT_METADATA_FILE
        self.workflow.priors_file = DEFAULT_PRIORS_FILE
        self.workflow.gold_standard_file = DEFAULT_GOLDSTANDARD_FILE
        self.workflow.expression_matrix_columns_are_genes = False

    def tearDown(self):
        del self.workflow

    def test_load_expression(self):
        self.workflow.read_expression()
        self.assertEqual(self.workflow.data.shape, (421, 100))
        np.testing.assert_allclose(np.sum(self.workflow.data.expression_data), 13507.22145160)

    def test_load_tf_names(self):
        self.workflow.read_tfs()
        self.assertEqual(len(self.workflow.tf_names), 100)
        tf_names = list(map(lambda x: "G" + str(x), list(range(1, 101))))
        self.assertListEqual(self.workflow.tf_names, tf_names)

    def test_load_priors_gs(self):
        self.workflow.read_priors()
        self.assertEqual(self.workflow.priors_data.shape, (100, 100))
        self.assertEqual(self.workflow.gold_standard.shape, (100, 100))
        self.assertTrue(all(self.workflow.priors_data.index == self.workflow.priors_data.columns))
        self.assertTrue(all(self.workflow.gold_standard.index == self.workflow.gold_standard.columns))

        self.workflow.priors_file = None
        self.workflow.priors_data = None
        self.workflow.gold_standard_file = None
        self.workflow.gold_standard = None

        with self.assertRaises(ValueError):
            self.workflow.validate_data()

    def test_load_metadata(self):
        self.workflow.read_expression()
        self.assertEqual(self.workflow.data.meta_data.shape, (421, 5))
        meta_data = pd.read_csv(os.path.join(self.workflow.input_dir, "meta_data.tsv"), sep="\t")
        meta_data.index = meta_data['condName']
        meta_data.index.name = None
        pdt.assert_frame_equal(meta_data, self.workflow.data.meta_data)
        self.workflow.read_expression()
        pdt.assert_frame_equal(meta_data, self.workflow.data.meta_data)

    def test_make_metadata(self):
        self.workflow.meta_data_file = None
        self.workflow.read_expression()
        self.assertEqual(self.workflow.data.meta_data.shape, (421, 4))

    def test_extract_metadata(self):
        self.workflow.read_expression()
        tmpdir = tempfile.TemporaryDirectory()

        merged_file = os.path.abspath(os.path.join(tmpdir.name, "expression.tsv"))
        meta_data = MetadataParserBranching.create_default_meta_data(self.workflow.data.sample_names)
        gene_list = self.workflow.data.gene_names
        test_data = pd.concat([self.workflow.data.to_df(), meta_data], axis=1)
        test_data.to_csv(merged_file, sep="\t")
        self.workflow.expression_matrix_file = merged_file
        self.workflow.meta_data_file = None
        self.workflow.expression_matrix_metadata = meta_data.columns.tolist()
        self.workflow.expression_matrix_columns_are_genes = True
        self.workflow.read_expression()

        pdt.assert_frame_equal(
            self.workflow.data.meta_data,
            MetadataParserBranching.fix_NAs(meta_data)
        )
        self.assertListEqual(
            self.workflow.data.gene_names.tolist(),
            gene_list.tolist()
        )

        tmpdir.cleanup()

    def test_load_gene_metadata(self):

        tempdir = tempfile.mkdtemp()

        try:
            gene_file = os.path.abspath(os.path.join(tempdir, "genes.tsv"))
            self.workflow.gene_metadata_file = gene_file
            self.workflow.gene_list_index = "SystematicName"
            genes = pd.DataFrame({"SystematicName": ["G1", "G2", "G3", "G4", "G7", "G6"]})
            genes.to_csv(gene_file, sep="\t", index=False)
            genes.index = genes['SystematicName']

            self.workflow.read_expression()

            pdt.assert_frame_equal(self.workflow.data.gene_data, genes.reindex(self.workflow.data.gene_names))
            self.assertListEqual(genes.index.tolist(), self.workflow.data.uns['trim_gene_list'].tolist())

            self.workflow.gene_list_index = None
            with self.assertRaises(ValueError):
                self.workflow.read_expression()

            self.workflow.gene_list_index = "SillyName"
            with self.assertRaises(ValueError):
                self.workflow.read_expression()
        finally:
            shutil.rmtree(tempdir)

    def test_get_data(self):
        self.workflow.get_data()
        self.assertTrue(self.workflow.data is not None)
        self.assertTrue(self.workflow.priors_data is not None)
        self.assertTrue(self.workflow.gold_standard is not None)
        self.assertTrue(self.workflow.tf_names is not None)

    def test_input_path(self):
        self.workflow.input_dir = None
        self.assertEqual(self.workflow.input_path("C"), os.path.abspath("C"))

        tempdir = tempfile.gettempdir()
        self.workflow.input_dir = tempdir
        self.assertEqual(self.workflow.input_path("A"), os.path.join(tempdir, "A"))

        absfile = os.path.join(os.path.abspath(os.sep), "B")
        self.assertEqual(self.workflow.input_path(absfile), absfile)

        with self.assertRaises(ValueError):
            self.assertIsNone(self.workflow.input_path(None))

    def test_null_network_generation(self):
        self.workflow.read_expression()
        self.workflow.read_tfs()
        self.assertIsNone(self.workflow.priors_data)
        self.assertIsNone(self.workflow.gold_standard)

        with self.assertRaises(ValueError):
            self.workflow.validate_data()

        self.workflow.use_no_gold_standard = True
        self.workflow.use_no_prior = True
        self.workflow.validate_data()

        self.assertIsNotNone(self.workflow.priors_data)
        self.assertListEqual(self.workflow.priors_data.columns.tolist(), self.workflow.tf_names)
        self.assertTrue(all(self.workflow.data.gene_names == self.workflow.priors_data.index))

        self.assertIsNotNone(self.workflow.gold_standard)
        self.assertIsNotNone(self.workflow.gold_standard)
        self.assertListEqual(self.workflow.gold_standard.columns.tolist(), self.workflow.tf_names)
        self.assertTrue(all(self.workflow.data.gene_names == self.workflow.gold_standard.index))

        self.workflow.gold_standard = None

        with self.assertWarns(Warning):
            self.workflow.use_no_gold_standard = True
            self.workflow.validate_data()

    def test_load_to_h5ad(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            dname = os.path.join(tmpdir, "dense.h5ad")
            sname = os.path.join(tmpdir, "sparse.h5ad")

            self.workflow.output_dir = tmpdir
            self.workflow.load_data_and_save_h5ad("dense.h5ad")

            data = ad.read_h5ad(dname)
            npt.assert_array_almost_equal_nulp(data.X, self.workflow.data.values)
            os.remove(dname)

            self.workflow.load_data_and_save_h5ad("sparse.h5ad", to_sparse=True)

            data = ad.read_h5ad(sname)
            self.assertTrue(sps.isspmatrix_csr(data.X))
            npt.assert_array_almost_equal_nulp(
                todense(data.X),
                todense(self.workflow.data.values)
            )
            os.remove(sname)


class TestWorkflowFunctions(unittest.TestCase):
    data = None

    @classmethod
    def setUpClass(cls):
        cls.data = workflow.WorkflowBase()
        cls.data.input_dir = os.path.join(my_dir, "../../data/dream4")
        cls.data.expression_matrix_file = "expression.tsv"
        cls.data.meta_data_file = "meta_data.tsv"
        cls.data.tf_names_file = "tf_names.tsv"
        cls.data.priors_file = "gold_standard.tsv"
        cls.data.gold_standard_file = "gold_standard.tsv"
        cls.data.expression_matrix_columns_are_genes = False
        cls.data.get_data()

    def setUp(self):
        self.workflow = workflow.WorkflowBase()
        self.workflow.priors_data = self.data.priors_data.copy()
        self.workflow.gold_standard = self.data.gold_standard.copy()
        self.workflow.data = self.data.data.copy()
        self.workflow.tf_names = self.data.tf_names
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.workflow.expression_matrix_file = "expression.tsv"
        self.workflow.expression_matrix_columns_are_genes = False

    def test_multiprocessing_init(self):
        MPControl.shutdown()
        self.workflow.multiprocessing_controller = "local"
        self.workflow.initialize_multiprocessing()
        self.assertTrue(MPControl.status())

    def test_abstractness(self):
        with self.assertRaises(NotImplementedError):
            self.workflow.startup()
        with self.assertRaises(NotImplementedError):
            self.workflow.startup_run()
        with self.assertRaises(NotImplementedError):
            self.workflow.startup_finish()
        with self.assertRaises(NotImplementedError):
            self.workflow.run()
        with self.assertRaises(NotImplementedError):
            self.workflow.emit_results(None, None, None, None)

    def test_append_path(self):
        self.workflow.append_to_path('input_dir', 'test')
        self.assertEqual(os.path.join(my_dir, "../../data/dream4", 'test'), self.workflow.input_dir)
        self.workflow.input_dir = None
        with self.assertRaises(ValueError):
            self.workflow.append_to_path('input_dir', 'test')

    def test_make_fake_metadata(self):
        self.workflow.data = None
        self.workflow.meta_data_file = None
        self.workflow.read_expression()
        self.assertEqual(self.workflow.data.meta_data.shape, (421, 4))

    def test_workflow_cv_priors_genes(self):
        self.workflow.split_gold_standard_for_crossvalidation = True
        self.workflow.cv_split_ratio = 0.5
        self.workflow.cv_split_axis = 0
        self.workflow.process_priors_and_gold_standard()
        self.assertEqual(self.workflow.priors_data.shape, (50, 100))
        self.assertEqual(self.workflow.gold_standard.shape, (50, 100))
        self.assertListEqual(self.workflow.priors_data.columns.tolist(), self.workflow.gold_standard.columns.tolist())
        self.workflow.align_priors_and_expression()
        self.assertEqual(self.workflow.priors_data.shape, (100, 100))
        self.assertEqual(self.workflow.gold_standard.shape, (50, 100))

    def test_workflow_cv_priors_tfs(self):
        self.workflow.split_gold_standard_for_crossvalidation = True
        self.workflow.cv_split_ratio = 0.5
        self.workflow.cv_split_axis = 1

        with self.assertWarns(UserWarning):
            self.workflow.process_priors_and_gold_standard()

        self.assertEqual(self.workflow.priors_data.shape, (100, 50))
        self.assertEqual(self.workflow.gold_standard.shape, (100, 50))
        self.assertListEqual(self.workflow.priors_data.index.tolist(), self.workflow.gold_standard.index.tolist())
        self.workflow.align_priors_and_expression()
        self.assertEqual(self.workflow.priors_data.shape, (100, 50))
        self.assertEqual(self.workflow.gold_standard.shape, (100, 50))

    def test_workflow_cv_priors_flat(self):
        self.workflow.split_gold_standard_for_crossvalidation = True
        self.workflow.cv_split_ratio = 0.5
        self.workflow.cv_split_axis = None

        with self.assertWarns(UserWarning):
            self.workflow.process_priors_and_gold_standard()

        self.assertEqual(self.workflow.priors_data.shape, (100, 100))
        self.workflow.align_priors_and_expression()
        self.assertEqual(self.workflow.priors_data.shape, (100, 100))

    def test_workflow_priors_filter(self):
        self.workflow.split_gold_standard_for_crossvalidation = True
        self.workflow.cv_split_ratio = 0.5
        self.workflow.cv_split_axis = 0
        self.workflow.tf_names = list(map(lambda x: "G" + str(x), list(range(1, 21))))
        self.workflow.gene_names = list(map(lambda x: "G" + str(x), list(range(1, 51))))
        self.workflow.read_expression()
        self.workflow.process_priors_and_gold_standard()

        self.assertEqual(self.workflow.gold_standard.shape, (50, 100))
        self.assertListEqual(self.workflow.priors_data.columns.tolist(), self.workflow.tf_names)

        self.workflow.align_priors_and_expression()
        self.assertEqual(self.workflow.priors_data.shape, (50, 20))
        self.assertEqual(self.workflow.data.shape, (421, 50))
        self.assertListEqual(self.workflow.priors_data.index.tolist(), self.workflow.data.gene_names.to_list())

    def test_get_bootstraps(self):
        bootstrap_0 = [37, 235, 396, 72, 255, 393, 203, 133, 335, 144, 129, 71, 237, 390, 281, 178, 276, 254, 357, 402,
                       395, 252, 156, 413, 398, 50, 68, 215, 241, 352, 86, 141, 393, 7, 319, 317, 22, 313, 1, 384, 316,
                       209, 264, 216, 141, 115, 121, 30, 71, 387, 405, 49, 313, 3, 280, 43, 76, 26, 308, 336, 109, 371,
                       297, 15, 64, 196, 25, 367, 226, 391, 282, 153, 104, 22, 265, 195, 126, 279, 381, 356, 155, 313,
                       83, 166, 136, 288, 418, 266, 279, 143, 239, 87, 281, 243, 348, 74, 190, 302, 416, 216, 151, 183,
                       321, 369, 333, 259, 384, 253, 262, 52, 2, 76, 149, 203, 263, 77, 200, 75, 332, 43, 20, 30, 36,
                       359, 263, 301, 57, 240, 210, 96, 269, 10, 279, 380, 337, 391, 377, 152, 202, 148, 416, 140, 193,
                       94, 60, 152, 338, 371, 353, 130, 220, 103, 354, 266, 182, 352, 338, 198, 194, 327, 176, 54, 15,
                       389, 401, 170, 20, 118, 278, 397, 114, 97, 181, 340, 10, 96, 183, 317, 56, 217, 405, 231, 96, 25,
                       398, 141, 212, 116, 299, 134, 205, 184, 399, 24, 137, 199, 309, 325, 420, 357, 248, 21, 296, 77,
                       219, 177, 369, 303, 45, 343, 144, 412, 234, 45, 372, 322, 302, 384, 413, 63, 331, 35, 33, 130,
                       83, 48, 310, 288, 253, 156, 55, 210, 287, 28, 222, 330, 136, 109, 99, 32, 8, 84, 50, 79, 169,
                       320, 108, 211, 24, 113, 276, 44, 271, 158, 398, 275, 251, 154, 235, 86, 391, 227, 53, 366, 243,
                       290, 100, 228, 288, 403, 280, 211, 229, 94, 166, 175, 231, 389, 79, 63, 369, 87, 416, 298, 202,
                       194, 216, 226, 158, 145, 324, 320, 188, 206, 145, 167, 163, 156, 150, 294, 169, 205, 326, 153,
                       230, 240, 48, 178, 300, 105, 182, 256, 342, 272, 275, 265, 112, 220, 51, 367, 138, 123, 324, 407,
                       270, 191, 21, 174, 380, 131, 344, 208, 54, 71, 14, 205, 143, 409, 309, 212, 114, 238, 413, 44,
                       37, 150, 332, 12, 376, 315, 410, 215, 125, 43, 370, 294, 91, 112, 253, 325, 226, 417, 171, 410,
                       184, 73, 308, 89, 27, 43, 1, 338, 127, 94, 410, 340, 30, 359, 64, 150, 98, 308, 131, 70, 140,
                       295, 230, 83, 239, 176, 317, 269, 164, 279, 406, 122, 249, 351, 53, 393, 169, 344, 365, 246, 221,
                       244, 204, 338, 362, 395, 105, 36, 112, 144, 158, 115, 106, 212, 291, 337, 258]

        self.workflow.response = self.workflow.data
        self.workflow.random_seed = 1
        self.workflow.num_bootstraps = 5
        bootstraps = self.workflow.get_bootstraps()
        self.assertEqual(len(bootstraps), 5)
        self.assertListEqual(bootstraps[0], bootstrap_0)

    def test_make_output_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.workflow.input_dir = temp_dir
        self.workflow.create_output_dir()
        self.assertTrue(os.path.exists(self.workflow.output_dir))
        os.rmdir(self.workflow.output_dir)
        os.rmdir(temp_dir)

    def test_shuffle_prior_labels(self):
        self.workflow.shuffle_prior_axis = 0
        np.testing.assert_array_almost_equal_nulp(self.workflow.priors_data.values, self.workflow.gold_standard.values)
        self.workflow.process_priors_and_gold_standard()
        self.assertTrue(all(self.workflow.priors_data.columns == self.workflow.gold_standard.columns))
        self.assertTrue(all(self.workflow.priors_data.index == self.workflow.gold_standard.index))
        self.assertTrue(all(self.workflow.priors_data.sum(axis=0) == self.workflow.gold_standard.sum(axis=0)))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal_nulp(self.workflow.priors_data.values,
                                                      self.workflow.gold_standard.values)

    def test_shuffle_prior_labels_2(self):
        self.workflow.shuffle_prior_axis = 1
        np.testing.assert_array_almost_equal_nulp(self.workflow.priors_data.values, self.workflow.gold_standard.values)
        self.workflow.process_priors_and_gold_standard()
        self.assertTrue(all(self.workflow.priors_data.columns == self.workflow.gold_standard.columns))
        self.assertTrue(all(self.workflow.priors_data.index == self.workflow.gold_standard.index))
        self.assertTrue(all(self.workflow.priors_data.sum(axis=1) == self.workflow.gold_standard.sum(axis=1)))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal_nulp(self.workflow.priors_data.values,
                                                      self.workflow.gold_standard.values)

    def test_add_prior_noise(self):
        self.workflow.set_shuffle_parameters(add_prior_noise=0.1)
        np.testing.assert_array_almost_equal_nulp(self.workflow.priors_data.values, self.workflow.gold_standard.values)
        self.workflow.process_priors_and_gold_standard()

        self.assertTrue(all(self.workflow.priors_data.columns == self.workflow.gold_standard.columns))
        self.assertTrue(all(self.workflow.priors_data.index == self.workflow.gold_standard.index))
        self.assertTrue(self.workflow.priors_data.sum().sum() > self.workflow.gold_standard.sum().sum())
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal_nulp(self.workflow.priors_data.values,
                                                      self.workflow.gold_standard.values)

class TestWorkflowFactory(unittest.TestCase):

    def test_base(self):
        worker = workflow.inferelator_workflow(regression=None, workflow=workflow.WorkflowBase)
        with self.assertRaises(NotImplementedError):
            worker.run()

    def test_bbsr(self):
        from inferelator.regression.bbsr_python import BBSRRegressionWorkflowMixin
        worker = workflow.inferelator_workflow(regression="bbsr", workflow=workflow.WorkflowBase)
        self.assertTrue(isinstance(worker, BBSRRegressionWorkflowMixin))

    def test_elasticnet(self):
        from inferelator.regression.elasticnet_python import ElasticNetWorkflowMixin
        worker = workflow.inferelator_workflow(regression="elasticnet", workflow=workflow.WorkflowBase)
        self.assertTrue(isinstance(worker, ElasticNetWorkflowMixin))

    def test_amusr(self):
        from inferelator.regression.amusr_regression import AMUSRRegressionWorkflowMixin
        from inferelator.workflows.amusr_workflow import MultitaskLearningWorkflow
        worker = workflow.inferelator_workflow(regression="amusr", workflow="amusr")
        self.assertTrue(isinstance(worker, AMUSRRegressionWorkflowMixin))
        self.assertTrue(isinstance(worker, MultitaskLearningWorkflow))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            worker = workflow.inferelator_workflow(regression="restlne", workflow=workflow.WorkflowBase)

        with self.assertRaises(ValueError):
            worker = workflow.inferelator_workflow(regression=1, workflow=workflow.WorkflowBase)

        with self.assertRaises(ValueError):
            worker = workflow.inferelator_workflow(regression=_RegressionWorkflowMixin, workflow="restlne")

        with self.assertRaises(ValueError):
            worker = workflow.inferelator_workflow(regression=_RegressionWorkflowMixin, workflow=None)
            
        with self.assertRaises(ValueError):
            worker = workflow.inferelator_workflow(regression=_RegressionWorkflowMixin, workflow=1)

    def test_tfa(self):
        from inferelator.workflows.tfa_workflow import TFAWorkFlow
        worker = workflow.inferelator_workflow(regression=_RegressionWorkflowMixin, workflow="tfa")
        self.assertTrue(isinstance(worker, TFAWorkFlow))

    def test_singlecell(self):
        from inferelator.workflows.single_cell_workflow import SingleCellWorkflow
        worker = workflow.inferelator_workflow(regression=_RegressionWorkflowMixin, workflow="single-cell")
        self.assertTrue(isinstance(worker, SingleCellWorkflow))
