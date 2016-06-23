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

    def test_separate_time_series(self):
        m_dicts = self.test_metadata_dicts()
        f = StringIO(expression_data_1)
        conditions = utils.conditions_from_tsv(f)
        (ts_dict, c_dict) = utils.separate_time_series(m_dicts, conditions)
        self.assertEqual({"wt", "c1", "c3", "c5"}, set(c_dict.keys()))
        assert len(ts_dict) == 1
        ts = ts_dict["c4"]
        name_order = ts.get_condition_name_order()
        self.assertEqual(["c4", "c2"], name_order)

    def test_metadata_df(self):
        f = StringIO(metadata_text_1)
        df = utils.metadata_df(f)
        self.assertEqual({'del.t', 'is1stLast', 'isTs', 'prevCol'}, set(df.keys()))
        return df

    def test_metadata_dicts(self):
        df = self.test_metadata_df()
        dicts = utils.metadata_dicts(df)
        expect = {
            'wt': {'del.t': False, 'nextCol': None, 'isTs': False, 'prevCol': False, 'is1stLast': 'e'}, 
            'c3': {'del.t': False, 'nextCol': None, 'isTs': False, 'prevCol': False, 'is1stLast': 'e'}, 
            'c2': {'del.t': 5, 'nextCol': None, 'isTs': False, 'prevCol': 'c4', 'is1stLast': 'l'}, 
            'c1': {'del.t': False, 'nextCol': None, 'isTs': False, 'prevCol': False, 'is1stLast': 'e'}, 
            'c5': {'del.t': False, 'nextCol': None, 'isTs': False, 'prevCol': False, 'is1stLast': 'e'}, 
            'c4': {'del.t': False, 'nextCol': 'c2', 'isTs': False, 'prevCol': False, 'is1stLast': 'f'}
        }
        for name in dicts:
            d = dicts[name]
            e = expect[name]
            self.assertEqual(d, e)
        self.assertEqual(dicts, expect)
        return dicts

    def test_conditions_from_tsv(self):
        text = u"" + (
            "\tcond1\tcond2\n"
            "gene1\t1\t2\n"
            "gene2\t3\t4\n"
            "gene3\t5\t4\n"
            )
        f = StringIO(text)
        conditions = utils.conditions_from_tsv(f)
        self.assertEqual({"cond1", "cond2"}, set(conditions.keys()))
        cond1 = conditions["cond1"]
        cgm = cond1.gene_mapping
        self.assertEqual({"gene1": 1, "gene2": 3, "gene3":5}, cgm.to_dict())

