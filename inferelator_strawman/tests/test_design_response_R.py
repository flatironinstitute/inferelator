import unittest, os
import pandas as pd
import numpy as np
from .. import design_response_R

my_dir = os.path.dirname(__file__)


class TestDRAboveDeltMax(unittest.TestCase):

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

    def calculate_design_and_response(self):
        drd = design_response_R.DR_driver()
        drd.target_directory = os.path.join(my_dir, "artifacts")
        drd.delTmin = self.delT_min
        drd.delTmax = self.delT_max
        drd.tau = self.tau
        (self.design, self.response) = drd.run(self.exp, self.meta)

    def test_design_matrix_above_delt_max(self):
        # Set up variables 
        ds, resp = (self.design, self.response)
        self.assertEqual(ds.shape, (2, 4))
        print "columns"
        print ds.columns
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

class TestDRR(unittest.TestCase):

    def xtest_save_R_driver(self):
        outfile = "artifacts/test_R_driver.R"
        outpath = os.path.join(my_dir, outfile)
        design_response_R.save_R_driver(outpath, tau=101)
        assert os.path.exists(outpath)
        text = open(outpath).read()
        assert "tau <- 101" in text
        # XXXX could clean up artifact, for now useful for debugging

    def xtest_DR_driver(self):
        drd = design_response_R.DR_driver()
        drd.target_directory = os.path.join(my_dir, "artifacts")
        meta = pd.DataFrame()
        meta['isTs']=[True, True, True, True, False]
        meta['is1stLast'] = ['f','m','m','l','e']
        meta['prevCol'] = ['NA','ts1','ts2','ts3', 'NA']
        meta['del.t'] = ['NA', 3, 2, 5, 'NA']
        meta['condName'] = ['ts1','ts2','ts3','ts4','ss']

        exp = pd.DataFrame(np.reshape(range(10), (2,5)) + 1,
         index = ['gene' + str(i + 1) for i in range(2)],
         columns = ['ts' + str(i + 1) for i in range(4)] + ['ss'])
        drd.delT_min = 2
        drd.delT_max = 4
        drd.tau = 2
        (design, response) = drd.run(exp, meta)
        print ("design")
        print (design)
        print ("response")
        print (response)
