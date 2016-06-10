
import unittest
from .. import condition
from .. import time_series
from .. import gene_model
import pandas as pd

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
