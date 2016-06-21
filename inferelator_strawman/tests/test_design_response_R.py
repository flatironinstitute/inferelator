import unittest, os
import pandas as pd
import numpy as np
from .. import design_response_R

my_dir = os.path.dirname(__file__)

class TestDRR(unittest.TestCase):

    def test_save_R_driver(self):
        outfile = "artifacts/test_R_driver.R"
        outpath = os.path.join(my_dir, outfile)
        design_response_R.save_R_driver(outpath, tau=101)
        assert os.path.exists(outpath)
        text = open(outpath).read()
        assert "tau <- 101" in text
        # XXXX could clean up artifact, for now useful for debugging

    def test_DR_driver(self):
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
