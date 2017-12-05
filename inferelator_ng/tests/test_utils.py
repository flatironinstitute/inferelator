import unittest
from .. import utils
from io import StringIO

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
