import unittest
from inferelator_ng import utils
from inferelator_ng.utils import Validator as check
from io import StringIO
import pandas as pd

metadata_text_1 = u"""
"isTs"\t"is1stLast"\t"prevCol"\t"del.t"\t"condName"
FALSE\t"e"\tNA\tNA\t"wt"
FALSE\t"e"\tNA\tNA\t"c1"
FALSE\t"l"\t"c4"\t5\t"c2"
FALSE\t"e"\tNA\tNA\t"c3"
FALSE\t"f"\tNA\tNA\t"c4"
FALSE\t"e"\tNA\tNA\t"c5"
""".strip()

expression_data_1 = u"""
\t"wt"\t"c1"\t"c2"\t"c3"\t"c4"\t"c5"
gene1\t1\t1\t1\t1\t1\t1
gene2\t0\t0\t0\t0\t0\t0
""".strip()

class TestUtils(unittest.TestCase):

    def test_read_tf_names(self):
        text = u'"G1"\n"G2"\n"G3"\n'
        f = StringIO(text)
        tf_names = utils.read_tf_names(f)
        self.assertEqual(["G1", "G2", "G3"], tf_names)

    def test_metadata_df(self):
        f = StringIO(metadata_text_1)
        df = utils.metadata_df(f)
        self.assertEqual({'del.t', 'is1stLast', 'isTs', 'prevCol'}, set(df.keys()))
        return df

class TestValidator(unittest.TestCase):

    def setUp(self):
        self.frame1 = pd.DataFrame(index=["A", "B", "C", "D", "E"], columns = ["RED", "BLUE", "GREEN"])
        self.frame2 = pd.DataFrame(index=["A", "B", "C", "D", "E"], columns = ["CYAN", "BLUE", "MAUVE"])
        self.frame3 = pd.DataFrame(index=["A", "B", "C", "E", "D"], columns = ["RED", "BLUE", "GREEN"])

    def test_frame_alignment(self):

        self.assertTrue(check.dataframes_align([self.frame1, self.frame1, self.frame1]))
        self.assertTrue(check.dataframes_align([self.frame1, self.frame1, self.frame3], check_order=False))

        with self.assertRaises(ValueError):
            check.dataframes_align([self.frame1, self.frame2, self.frame1])

        with self.assertRaises(ValueError):
            check.dataframes_align([self.frame1, self.frame3, self.frame1])

    def test_numeric(self):

        self.assertTrue(check.argument_numeric(0))
        self.assertTrue(check.argument_numeric(0.0))

        with self.assertRaises(ValueError):
            check.argument_numeric("0")

        self.assertTrue(check.argument_numeric(1, 0, 2))

        with self.assertRaises(ValueError):
            self.assertTrue(check.argument_numeric(2, 0, 1))

        self.assertTrue(check.argument_numeric(None, allow_none=True))

    def test_type(self):

        self.assertTrue(check.argument_type(self, unittest.TestCase))
        self.assertTrue(check.argument_type(None, unittest.TestCase, allow_none=True))

        with self.assertRaises(ValueError):
            self.assertTrue(check.argument_type("0", unittest.TestCase))

    def test_enum(self):

        self.assertTrue(check.argument_enum("A", ("A", "B")))
        self.assertTrue(check.argument_enum(["A", "B", "A"], ("A", "B")))

        with self.assertRaises(ValueError):
            check.argument_enum(["A", "B", "C"], ("A", "B"))

    def test_none(self):

        self.assertTrue(check.arguments_not_none(("A", "B")))
        self.assertTrue(check.arguments_not_none(("A", None), num_none=1))
        with self.assertRaises(ValueError):
            self.assertTrue(check.arguments_not_none((None, None, "A")))
        with self.assertRaises(ValueError):
            self.assertTrue(check.arguments_not_none((None, None, "A"), num_none=0))

if __name__ == '__main__':
    unittest.main()



