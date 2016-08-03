import unittest
from .. import tfa
import pandas as pd
import numpy as np
import subprocess

class TestTFA(unittest.TestCase):

    def generate_random_matrix(self, n, m):
        return np.array([np.random.rand(m) for x in range(n)])

    # Test for 5 genes, one of which is a TF, 5 condidtions, and 4 TFs.
    # where tau is equal to 1, so exp_mat and exp_mat_tau are equivalent
    def setup(self):
        exp = pd.DataFrame(self.generate_random_matrix(5, 5))
        exp.columns = ['s1', 's2', 's3', 's4', 's5']
        exp.index = ['g1', 't2', 'g3', 'g4', 'g5']
        priors = pd.DataFrame(np.array([[1,0,0,1], [0,0,0,0], [0,0,-1,0], [-1,0,0,-1], [0,0,1,0]]))
        priors.columns = ['t1', 't2', 't3', 't4']
        priors.index = ['g1', 't2', 'g3', 'g4', 'g5']
        self.tfa_python = tfa.TFA(priors, exp, exp)

    # def test_three_by_two(self):
    #     self.setup()
    #     activities = self.tfa_class.tfa()
    #     print activities

    # def test_three_by_two_dupes(self):
    #     self.setup()
    #     activities = self.tfa_class.tfa(dup_self = False)
    #     print activities

    # def test_three_by_two_self_interactions(self):
    #     self.setup()
    #     activities = self.tfa_class.tfa(noself = False)
    #     print activities

    # def test_three_by_two_self_interactions_and_dupes(self):
    #     self.setup()
    #     activities = self.tfa_class.tfa(noself = False, dup_self = False)
    #     print activities
    def test_tfa_default(self):
        self.setup()
        activities = self.tfa_class.tfa()
        print activities