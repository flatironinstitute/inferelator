
import unittest
from .. import condition
import pandas as pd

class TestCondition(unittest.TestCase):

    def test_convert_dictionary(self):
        "Test that a dictionary works as condition data."
        D = {"gene1": 3}
        c = condition.Condition("my condition", D)
        self.assertEqual(c.name, "my condition")
        m = c.gene_mapping
        self.assertEqual(type(m), pd.Series)
        self.assertEqual(m.to_dict(), D)

    def test_cant_convert(self):
        with self.assertRaises(Exception):
            c = condition.Condition("bad condition", pd.Series)
