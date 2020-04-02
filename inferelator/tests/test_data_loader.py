import unittest
import numpy.testing as npt
import bio_test_artifacts.prebuilt as test_prebuilt
from inferelator.workflow import inferelator_workflow


class TestExpressionLoader(unittest.TestCase):

    def setUp(self):

        self.worker = inferelator_workflow()

    def test_tsv(self):

        file, data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='tsv')
        data.to_csv(file, sep="\t")

        self.worker.set_expression_file(tsv=file)
        self.worker.read_expression()

        npt.assert_array_almost_equal(data.values, self.worker.data.expression_data)

    def test_h5ad(self):
        file, data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='h5ad')

        self.worker.set_expression_file(h5ad=file)
        self.worker.read_expression()

        npt.assert_array_almost_equal(data.values, self.worker.data.expression_data)

    def test_hdf5(self):
        file, data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='hdf5')

        self.worker.set_expression_file(hdf5=file)
        self.worker.read_expression()

        npt.assert_array_almost_equal(data.values, self.worker.data.expression_data)

    def test_mtx(self):
        (file1, file2, file3), data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='mtx')

        self.worker.set_expression_file(mtx=file1, mtx_feature=file2, mtx_barcode=file3)
        self.worker.read_expression()

        npt.assert_array_almost_equal(data.values, self.worker.data.expression_data.A)