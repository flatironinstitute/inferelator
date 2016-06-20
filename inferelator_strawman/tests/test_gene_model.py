
import unittest
from .. import condition
from .. import time_series
from .. import gene_model
import pandas as pd
import numpy as np

class TestGeneModel(unittest.TestCase):

    def test_conditions_design_response(self):
        c1 = condition.Condition("c1", {"tf": 1, "g1": 2, "g2": 3, "g3": 4})
        c2 = condition.Condition("c2", {"tf": 5, "g1": 6, "g2": 7, "g3": 8})
        genes = ["g1", "g2", "g3"]
        tfs = ["tf"]
        tr = time_series.TransitionResponse(tau_half_life=2.0)
        conditions = [c1, c2]
        model = gene_model.GeneModel(genes, tfs, tr)
        got_response = model.response_matrix(conditions).tolist()
        expected_response = [
            [2.0, 3.0, 4.0], 
            [6.0, 7.0, 8.0]]
        self.assertEqual(got_response, expected_response)
        got_design = model.design_matrix(conditions).tolist()
        expect_design = [
            [1.0], 
            [5.0]]
        self.assertEqual(got_design, expect_design)

    def test_time_series_design_response(self):
        first = condition.Condition("first", {"tf": 1, "g1": 2, "g2": 2, "g3": 2})
        second = condition.Condition("second", {"tf": 5, "g1": 6, "g2": 7, "g3": 8})
        genes = ["g1", "g2", "g3"]
        tfs = ["tf"]
        tr = time_series.TransitionResponse(tau_half_life=2.0)
        model = gene_model.GeneModel(genes, tfs, tr)
        ts = time_series.TimeSeries(first)
        ts.add_condition("first", second, 2)
        got_response = model.response_matrix_ts(ts).tolist()
        expect_response = [
            [ 2.,  2.,  2.],
            [ 5.,  6., 7.]]
        self.assertEqual(got_response, expect_response)
        got_design = model.design_matrix_ts(ts).tolist()
        expect_design = [[1.0], [5.0]]
        self.assertEqual(got_design, expect_design)

    def test_design_and_response(self):
        c1 = condition.Condition("c1", {"tf": 11, "g1": 2, "g2": 3, "g3": 4})
        c2 = condition.Condition("c2", {"tf": 55, "g1": 6, "g2": 7, "g3": 8})
        first = condition.Condition("first", {"tf": 1, "g1": 2, "g2": 2, "g3": 2})
        second = condition.Condition("second", {"tf": 5, "g1": 6, "g2": 7, "g3": 8})
        genes = ["g1", "g2", "g3"]
        tfs = ["tf"]
        tr = time_series.TransitionResponse(tau_half_life=2.0)
        ts = time_series.TimeSeries(first)
        ts.add_condition("first", second, 2)
        model = gene_model.GeneModel(genes, tfs, tr)
        dr = model.design_and_response([c1, c2], [ts])
        self.assertEqual([c1, c2, first, second], dr.all_conditions)
        self.assertEqual([[11], [55], [1], [5]], dr.design.tolist())
        expect_response = [
            [ 2,  3,  4],
            [ 6,  7,  8],
            [ 2,  2,  2],
            [ 5,  6,  7]]
        self.assertEqual(expect_response, dr.response.tolist())
        tsv = model.meta_data_tsv([c1, c2], [ts])
        expect = (
            'isTs\tis1stLast\tprevCol\tdel.t\tcondName\n'
            'False\t"e"\tNA\tNA\t"c1"\n'
            'False\t"e"\tNA\tNA\t"c2"\n'
            'True\t"f"\tNA\tNA\t"first"\n'
            'True\t"l"\t"first"\t2\t"second"\n')
        self.assertEqual(tsv, expect)
        df = model.expression_data_frame([c1, c2], [ts])
        self.assertEqual(['c1', 'c2', 'first', 'second'], list(df.columns))
        self.assertEqual(['tf', 'g1', 'g2', 'g3'], list(df.index))
