import unittest
import shutil
import os
import tempfile
import pandas as pd
import anndata as ad
import numpy as np
import numpy.testing as npt
import pandas.testing as pdt
import bio_test_artifacts.prebuilt as test_prebuilt
from inferelator.workflow import inferelator_workflow
from inferelator.utils import loader, todense


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

    def test_h5ad_obj(self):
        _, data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='h5ad')

        self.worker.set_expression_file(h5ad=ad.AnnData(data))
        self.worker.read_expression()

        npt.assert_array_almost_equal(data.values, self.worker.data.expression_data)

    @unittest.skip('numpy2/tables incompatibility')
    def test_hdf5(self):
        file, data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='hdf5')

        self.worker.set_expression_file(hdf5=file)
        self.worker.read_expression()

        npt.assert_array_almost_equal(data.values, self.worker.data.expression_data)

    def test_mtx(self):
        (file1, file2, file3), data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='mtx')

        self.worker.set_expression_file(mtx=file1, mtx_feature=file2, mtx_barcode=file3)
        self.worker.read_expression()

        npt.assert_array_almost_equal(
            data.values,
            todense(self.worker.data.expression_data)
        )

    def test_10x(self):
        (file1, file2, file3), data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='mtx')

        with tempfile.TemporaryDirectory() as txdir:
            shutil.copy(file1, os.path.join(txdir, "matrix.mtx"))
            shutil.copy(file2, os.path.join(txdir, "genes,tsv"))
            shutil.copy(file3, os.path.join(txdir, "barcodes.tsv"))

            self.worker.set_expression_file(tenx_path=txdir)
            self.worker.read_expression()

            npt.assert_array_almost_equal(
                data.values,
                todense(self.worker.data.expression_data)
            )

    def test_10x_ranger3(self):
        (file1, file2, file3), data = test_prebuilt.counts_yeast_single_cell_chr01(filetype='mtx', gzip=True)

        with tempfile.TemporaryDirectory() as txdir:
            shutil.copy(file1, os.path.join(txdir, "matrix.mtx.gz"))
            shutil.copy(file2, os.path.join(txdir, "features.tsv.gz"))
            shutil.copy(file3, os.path.join(txdir, "barcodes.tsv.gz"))

            self.worker.set_expression_file(tenx_path=txdir)
            self.worker.read_expression()

            npt.assert_array_almost_equal(
                data.values,
                todense(self.worker.data.expression_data)
            )

    def test_df_decode(self):
        idx = pd.Index(['str1', b'str2', b'str3', 'str4', 5, 17.4, np.inf, ('str1',)])
        correct = pd.Index(['str1', 'str2', 'str3', 'str4', 5, 17.4, np.inf, ('str1',)])

        df1 = pd.DataFrame(idx.tolist(), index=idx)
        df1_c = pd.DataFrame(correct.tolist(), index=correct)

        loader._safe_dataframe_decoder(df1)
        pdt.assert_frame_equal(df1, df1_c)

        vals = np.random.rand(len(idx), len(idx))
        df2 = pd.DataFrame(vals, index=idx, columns=idx)
        df2_c = pd.DataFrame(vals, index=correct, columns=correct)

        loader._safe_dataframe_decoder(df2)

        pdt.assert_frame_equal(df2, df2_c)

        df3 = pd.DataFrame(pd.Categorical(idx.tolist()), index=idx)
        df3_c = pd.DataFrame(pd.Categorical(correct.tolist()), index=correct)

        loader._safe_dataframe_decoder(df3)

        pdt.assert_frame_equal(df3, df3_c)
