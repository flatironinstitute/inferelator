import unittest, os
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
