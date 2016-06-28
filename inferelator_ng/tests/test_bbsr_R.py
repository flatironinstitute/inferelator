import unittest, os
import pandas as pd
import numpy as np
from .. import bbsr_R

my_dir = os.path.dirname(__file__)

class TestDR(unittest.TestCase):
    """
    Superclass for common methods
    """
    def test_calculate_design_and_response(self):
        brd = bbsr_R.BBSR_driver()
        brd.target_directory = os.path.join(my_dir, "artifacts")
        self.X = pd.DataFrame([1 ,2, 3])
        self.Y = pd.DataFrame([1])
        self.priors = pd.DataFrame([])
        self.clr = pd.DataFrame([1 ,2, 3])
        (i, j) = brd.run(self.X, self.Y, self.clr, self.priors)
        print i
        print j