import unittest
from .. import tfa
import pandas as pd
import numpy as np

class TestTFA(unittest.TestCase):

    def generate_random_matrix(self, n, m):
        return np.array([np.random.rand(m) for x in range(n)])

    # Test for 3 genes, 2 of which are TFs, and 2 condidtions
    # where tau is equal to 1, so exp_mat and exp_mat_tau are equivalent
    def setup(self):
        exp = pd.DataFrame(self.generate_random_matrix(2, 3))
        priors = pd.DataFrame(np.array([[0, 1, 0], [1, 0, 0]]).transpose())
        self.tfa_class = tfa.TFA(priors, exp, exp)

    def test_three_by_two(self):
        self.setup()
        activities = self.tfa_class.tfa()
        print activities

    def test_three_by_two_dupes(self):
        self.setup()
        activities = self.tfa_class.tfa(dup_self = False)
        print activities

    def test_three_by_two_self_interactions(self):
        self.setup()
        activities = self.tfa_class.tfa(noself = False)
        print activities

    def test_three_by_two_self_interactions_and_dupes(self):
        self.setup()
        activities = self.tfa_class.tfa(noself = False, dup_self = False)
        print activities