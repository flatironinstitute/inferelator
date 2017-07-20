import unittest, os
import pandas as pd
import numpy as np
import pdb
from .. import design_response_translation
from .. import utils

my_dir = os.path.dirname(__file__)

class TestDR(unittest.TestCase):
    """
    Superclass for common methods
    """
    def calculate_design_and_response(self):
        #drd = design_response_R.DRDriver()
        drd = design_response_translation.PythonDRDriver()
        target = drd.target_directory = os.path.join(my_dir, "artifacts")
        if not os.path.exists(target):
            os.makedirs(target)
        drd.delTmin = self.delT_min
        drd.delTmax = self.delT_max
        drd.tau = self.tau
        (self.design, self.response) = drd.run(self.exp, self.meta)

class TestSpecialCharacter(TestDR):

    def setUp(self):
        spchrs='~!@#$%^&*()_-+=|\}]{[;:/?.><\\'
        self.meta = pd.DataFrame()
        self.meta['isTs']=[True, True, True, True, False]
        self.meta['is1stLast'] = ['f','m','m','l','e']
        self.meta['prevCol'] = ['NA','ts1'+spchrs,'ts2'+spchrs,'ts3'+spchrs, 'NA']
        self.meta['del.t'] = ['NA', 3, 2, 5, 'NA']
        self.meta['condName'] = ['ts1'+spchrs,'ts2'+spchrs,'ts3'+spchrs,'ts4'+spchrs,'ss']
        self.exp = pd.DataFrame(np.reshape(range(10), (2,5)) + 1,
            index = ['gene' + str(i + 1) + spchrs for i in range(2)],
            columns = ['ts' + str(i + 1) + spchrs for i in range(4)] + ['ss'])
        self.delT_min = 2
        self.delT_max = 4
        self.tau = 2
        self.calculate_design_and_response()

    def testspecialcharacter(self):
        spchrs='~!@#$%^&*()_-+=|\}]{[;:/?.><\\'
        ds, resp = (self.design, self.response)
        expression_1 = np.array(list(self.exp['ts1' + spchrs]))
        expression_2 = np.array(list(self.exp['ts2' + spchrs]))
        expected_response_1 = (expression_1 + self.tau * (expression_2 - expression_1) / (float(self.meta['del.t'][1])))
        expression_3 = np.array(list(self.exp['ts3' + spchrs]))
        expected_response_2 = expression_2 + self.tau * (expression_3 - expression_2) /  (float(self.meta['del.t'][2]))
        np.testing.assert_almost_equal(np.array(resp['ts1' + spchrs]), expected_response_1)
        np.testing.assert_almost_equal(np.array(resp['ts2' + spchrs]), expected_response_2)


class TestDRModelOrganisms(TestDR):

    def test_on_bsubtilis(self):
        self.exp = utils.df_from_tsv('data/bsubtilis/expression.tsv')
        self.meta = utils.df_from_tsv('data/bsubtilis/meta_data.tsv', has_index=False)
        expected_design = utils.df_from_tsv('data/bsubtilis/bsubtilis_design_matrix.tsv')
        expected_response = utils.df_from_tsv('data/bsubtilis/bsubtilis_response_matrix.tsv')
        self.delT_min = 0
        self.delT_max = 110
        self.tau = 45
        self.calculate_design_and_response()
        np.testing.assert_allclose(self.response.values, expected_response.values, atol=1e-15)
        self.assertEqual(len(set(expected_response.columns)), len(set(self.response.columns)))
        self.assertEqual(expected_response.columns.tolist(), self.response.columns.tolist())
        self.assertEqual(expected_response.index.tolist(), self.response.index.tolist())
        self.assertTrue(pd.DataFrame.equals(expected_design, self.design))

class TestDRAboveDeltMax(TestDR):

    def setUp(self):
        self.meta = pd.DataFrame()
        self.meta['isTs']=[True, True, True, True, False]
        self.meta['is1stLast'] = ['f','m','m','l','e']
        self.meta['prevCol'] = ['NA','ts1','ts2','ts3', 'NA']
        self.meta['del.t'] = ['NA', 3, 2, 5, 'NA']
        self.meta['condName'] = ['ts1','ts2','ts3','ts4','ss']
        self.exp = pd.DataFrame(np.reshape(range(10), (2,5)) + 1,
            index = ['gene' + str(i + 1) for i in range(2)],
            columns = ['ts' + str(i + 1) for i in range(4)] + ['ss'])
        self.delT_min = 2
        self.delT_max = 4
        self.tau = 2
        self.calculate_design_and_response()

    def test_design_matrix_above_delt_max(self):
        # Set up variables
        ds, resp = (self.design, self.response)
        self.assertEqual(ds.shape, (2, 4))
        self.assertEqual(list(ds.columns), ['ts4', 'ss', 'ts1', 'ts2'],
            msg = "Guarantee that the ts3 condition is dropped, "
                  "since its delT of 5 is greater than delt_max of 4")
        for col in ds:
            self.assertEqual(list(ds[col]), list(self.exp[col]),
                msg = ('{} column in the design matrix should be equal '
                    'to that column in the expression matrix').format(col))

        self.assertEqual(list(ds['ss']), [5, 10])
        self.assertEqual(list(ds['ss']), list(resp['ss']),
            msg = 'Steady State design and response should be equal')
        self.assertTrue((resp['ts2'].values == [3, 8]).all())

    def test_response_matrix_steady_state_above_delt_max(self):
        ds, resp = (self.design, self.response)
        self.assertEqual(list(resp.columns), ['ts4', 'ss', 'ts1', 'ts2'])
        self.assertEqual(list(resp['ts4']), list(self.exp['ts4']))
        self.assertEqual(list(resp['ss']), list(self.exp['ss']))

    def test_response_matrix_time_series_above_delt_max(self):
        ds, resp = (self.design, self.response)
        expression_1 = np.array(list(self.exp['ts1']))
        expression_2 = np.array(list(self.exp['ts2']))
        expected_response_1 = (expression_1 + self.tau * (expression_2 - expression_1) / (
            float(self.meta['del.t'][1])))
        expression_3 = np.array(list(self.exp['ts3']))
        expected_response_2 = expression_2 + self.tau * (expression_3 - expression_2) /  (
            float(self.meta['del.t'][2]))
        np.testing.assert_almost_equal(np.array(resp['ts1']), expected_response_1)
        np.testing.assert_almost_equal(np.array(resp['ts2']), expected_response_2)

class TestDRMicro(TestDR):

    def setUp(self):
        self.meta = pd.DataFrame()
        self.meta['isTs']=[False, False]
        self.meta['is1stLast'] = ['e','e']
        self.meta['prevCol'] = ['NA','NA']
        self.meta['del.t'] = ['NA', 'NA']
        self.meta['condName'] = ['ss1', 'ss2']
        self.exp = pd.DataFrame(
            np.reshape(range(4), (2, 2)) + 1,
            index=['gene' + str(i + 1) for i in range(2)],
            columns=['ss1', 'ss2'])
        self.delT_min = 2
        self.delT_max = 4
        self.tau = 2
        self.calculate_design_and_response()

    def test_micro(self):
        ds, resp = (self.design, self.response)
        self.assertEqual(ds.shape, (2, 2))
        self.assertTrue((ds['ss1'].values == [1, 3]).all())
        self.assertTrue((ds['ss2'].values == [2, 4]).all())
        # In steady state, expect design and response to be identical
        self.assertTrue(ds.equals(resp))

class TestDRBelowDeltMin(TestDR):

    def setUp(self):
        self.meta = pd.DataFrame()
        self.meta['isTs']=[True, True, True, True, False]
        self.meta['is1stLast'] = ['f','m','m','l','e']
        self.meta['prevCol'] = ['NA','ts1','ts2','ts3', 'NA']
        self.meta['del.t'] = ['NA', 1, 2, 3, 'NA']
        self.meta['condName'] = ['ts1', 'ts2', 'ts3', 'ts4', 'ss']
        self.exp = pd.DataFrame(
            np.reshape(range(10), (2, 5)) + 1,
            index=['gene' + str(i + 1) for i in range(2)],
            columns=['ts' + str(i + 1) for i in range(4)] + ['ss'])
        self.delT_min = 2
        self.delT_max = 4
        self.tau = 2
        self.calculate_design_and_response()

    def test_response_matrix_below_delt_min(self):
        ds, resp = (self.design, self.response)
        expression_1 = np.array(list(self.exp['ts1']))
        expression_3 = np.array(list(self.exp['ts3']))
        expected_response_1 = expression_1 + self.tau * (expression_3 - expression_1) /  (float(self.meta['del.t'][1]) + float(self.meta['del.t'][2]))
        np.testing.assert_almost_equal(np.array(resp['ts1']), expected_response_1)
        #pdb.set_trace()

    @unittest.skip("skipping until we've determined if we want to modify the legacy R code")
    def test_design_matrix_headers_below_delt_min(self):
        ds, resp = (self.design, self.response)
        print(ds.columns)
        self.assertEqual(list(ds.columns), ['ss', 'ts1', 'ts2', 'ts3'],
            msg = "Guarantee that the ts4 condition is dropped, since its the last in the time series")

class TestBranchingTimeSeries(TestDR):

    def setUp(self):
        self.meta = pd.DataFrame()
        self.meta['isTs']=[True, True, True]
        self.meta['is1stLast'] = ['f','l','l']
        self.meta['prevCol'] = ['NA','ts1','ts1']
        self.meta['del.t'] = ['NA', 2, 2]
        self.meta['condName'] = ['ts1','ts2','ts3']
        self.exp = pd.DataFrame(np.reshape(range(9), (3,3)) + 1,
            index = ['gene' + str(i + 1) for i in range(3)],
            columns = ['ts' + str(i + 1) for i in range(3)])
        self.delT_min = 1
        self.delT_max = 4
        self.tau = 1
        self.calculate_design_and_response()

    def test_design_matrix_branching_time_series(self):
        ds, resp = (self.design, self.response)
        self.assertEqual(ds.shape, (3, 2))
        self.assertEqual(list(ds.columns), ['ts1_dupl01', 'ts1_dupl02'],
             msg = 'This is how the R code happens to name branching time series')
        for col in ds:
            self.assertEqual(list(ds[col]), list(self.exp['ts1']),
                msg = '{} column in the design matrix should be equal to the branching source, ts1, in the exp matrix'.format(col))

    def test_response_matrix_branching_time_series(self):
        ds, resp = (self.design, self.response)
        self.assertEqual(resp.shape, (3, 2))
        expression_1 = np.array(list(self.exp['ts1']))
        expression_2 = np.array(list(self.exp['ts2']))
        expected_response_1 = (expression_1 + self.tau * (expression_2 - expression_1) /
            float(self.meta['del.t'][1]))

        expression_3 = np.array(list(self.exp['ts3']))
        expected_response_2 = (expression_1 + self.tau * (expression_3 - expression_1) /
            float(self.meta['del.t'][2]))

        np.testing.assert_almost_equal(np.array(resp['ts1_dupl01']), expected_response_1)
        np.testing.assert_almost_equal(np.array(resp['ts1_dupl02']), expected_response_2)
