
import unittest
from .. import condition
import pandas as pd

class TestCondition(unittest.TestCase):

    def test_convert_dictionary(self):
        "Test that a dictionary works as condition data."
        D = {"gene1": 3, "tf1": 6, "tf2": -1}
        c = condition.Condition("my condition", D)
        self.assertEqual(c.name, "my condition")
        m = c.gene_mapping
        self.assertEqual(type(m), pd.Series)
        self.assertEqual(m.to_dict(), D)
        self.assertEqual(c.response_scalar("gene1"), D["gene1"])
        self.assertEqual(list(c.design_vector(["tf1", "tf2"])), [D["tf1"], D["tf2"]])
        tsv = c.meta_data_tsv_line()
        self.assertEqual('isTs\tis1stLast\tprevCol\tdel.t\tcondName\n', condition.Condition.META_DATA_HEADER)
        self.assertEqual('False\t"e"\tNA\tNA\t"my condition"\n', tsv)

    def test_cant_convert(self):
        with self.assertRaises(Exception):
            c = condition.Condition("bad condition", pd.Series)
