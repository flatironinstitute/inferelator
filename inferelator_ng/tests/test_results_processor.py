import unittest
from .. import results_processor
import pandas as pd

class TestResultsProcessor(unittest.TestCase):

    def test_combining_confidenses(self):
        print 'in test'