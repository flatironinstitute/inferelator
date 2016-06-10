import unittest
from .. import condition
from .. import time_series
import pandas as pd

class TestTimeSeries(unittest.TestCase):

    def test_1_condition(self):
        first = condition.Condition("first", {"gene1": 9, "gene2": 0.12})
        ts = time_series.TimeSeries(first)
        name_order = ts.get_condition_name_order()
        self.assertEqual([first.name], name_order)
        return ts

    def test_no_add_after_compile(self):
        second = condition.Condition("second", {"gene1": 6, "gene2": 3})
        ts = self.test_1_condition()
        with self.assertRaises(AssertionError):
            ts.add_condition("first", second, 12)

    def test_2_conditions(self):
        first = condition.Condition("first", {"gene1": 9, "gene2": 0.12})
        second = condition.Condition("second", {"gene1": 6, "gene2": 3})
        ts = time_series.TimeSeries(first)
        ts.add_condition("first", second, 12)
        name_order = ts.get_condition_name_order()
        self.assertEqual([first.name, second.name], name_order)
        interval_order = ts.get_interval_order()
        self.assertEqual([0, 12], interval_order)
        name_order_again =  ts.get_condition_name_order()
        assert name_order_again is name_order
        fg1r = ts.get_response_parameters("first", "gene1")
        self.assertEqual(None, fg1r.gene_level_before)
        self.assertEqual(9, fg1r.gene_level)
        self.assertEqual(0, fg1r.time_interval)
        sg1r = ts.get_response_parameters("second", "gene1")
        self.assertEqual(9, sg1r.gene_level_before)
        self.assertEqual(6, sg1r.gene_level)
        self.assertEqual(12, sg1r.time_interval)