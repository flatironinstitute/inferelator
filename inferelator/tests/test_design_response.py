import pytest
import os
import pandas as pd
import numpy as np
from inferelator.preprocessing import metadata_parser
from inferelator.preprocessing import design_response_translation
from inferelator import utils

my_dir = os.path.dirname(__file__)


class TestMetaDataProcessor:

    def setup_method(self):
        self.meta = pd.DataFrame({
            'isTs': [True, True, True, True, False],
            'is1stLast': ['f', 'm', 'm', 'l', 'e'],
            'prevCol': ['NA', 'ts1', 'ts2', 'ts3', 'NA'],
            'del.t': ['NA', 3, 2, 5, 'NA'],
            'condName': ['ts1', 'ts2', 'ts3', 'ts4', 'ss']
        })
        self.expr = pd.DataFrame(np.ones((10,5)), columns=self.meta['condName'])

    def test_NA_fix(self):
        meta = metadata_parser.MetadataParserBranching.fix_NAs(self.meta)
        assert pd.isnull(meta['del.t']).sum() == 2
        assert pd.isnull(meta['prevCol']).sum() == 2
        assert pd.isnull(meta['isTs']).sum() == 0
        assert pd.isnull(meta['condName']).sum() == 0

    def test_meta_processing_steady(self):
        meta = metadata_parser.MetadataParserBranching.fix_NAs(self.meta)
        steady_idx, ts_idx = metadata_parser.MetadataParserBranching.process_groups(meta)
        assert len(steady_idx.keys()) == 5
        assert sum(steady_idx.values()) == 1
        assert steady_idx["ss"]

    def test_meta_processing_time(self):
        meta = metadata_parser.MetadataParserBranching.fix_NAs(self.meta)
        steady_idx, ts_idx = metadata_parser.MetadataParserBranching.process_groups(meta)
        assert len(ts_idx.keys()) == 4
        assert ts_idx["ts1"] == [(None, None), ("ts2", 3)]
        assert ts_idx["ts2"] == [("ts1", 3), ("ts3", 2)]
        assert ts_idx["ts3"] == [("ts2", 2), ("ts4", 5)]
        assert ts_idx["ts4"] == [("ts3", 5), (None, None)]

    def test_checking_missing_samples(self):
        meta = metadata_parser.MetadataParserBranching.fix_NAs(self.meta)

        with pytest.raises(metadata_parser.ConditionDoesNotExistError):
            meta_err = meta.copy().iloc[0:2, :]
            steady_idx, ts_idx = metadata_parser.MetadataParserBranching.process_groups(meta_err)
            metadata_parser.MetadataParserBranching.check_for_dupes(self.expr, meta_err, steady_idx,
                                                                    strict_checking_for_metadata=True)

        meta_err = meta.copy().iloc[0:2, :]
        steady_idx, ts_idx = metadata_parser.MetadataParserBranching.process_groups(meta_err)
        new_idx = metadata_parser.MetadataParserBranching.check_for_dupes(self.expr, meta_err, steady_idx)
        assert sum(new_idx.values()) == 3

    def test_checking_dupe_samples(self):
        meta = metadata_parser.MetadataParserBranching.fix_NAs(self.meta)

        with pytest.raises(metadata_parser.MultipleConditionsError):
            meta_err = meta.copy()
            meta_err['condName'] = "allsame"
            steady_idx, ts_idx = metadata_parser.MetadataParserBranching.process_groups(meta_err)
            metadata_parser.MetadataParserBranching.check_for_dupes(self.expr, meta_err, steady_idx,
                                                                    strict_checking_for_duplicates=True)

        meta_err = meta.copy()
        meta_err['condName'] = "allsame"
        steady_idx, ts_idx = metadata_parser.MetadataParserBranching.process_groups(meta_err)
        metadata_parser.MetadataParserBranching.check_for_dupes(self.expr, meta_err, steady_idx,
                                                                strict_checking_for_duplicates=False)

    def test_fixing_alignment(self):
        meta = metadata_parser.MetadataParserBranching.fix_NAs(self.meta)
        meta = meta.copy()
        meta.index = meta['condName']
        del meta['condName']
        metadata_parser.MetadataParserBranching.validate_metadata(self.expr, meta)
        steady_idx, ts_idx = metadata_parser.MetadataParserBranching.process_groups(meta)
        assert len(steady_idx.keys()) == 5
        assert len(ts_idx.keys()) == 4


class TestMetaDataNonbranchingProcessor:

    def setup_method(self):
        self.meta = pd.DataFrame({
            'strain': ['a', 'a', 'a', 'a', 'b'],
            'time': [0, 3, 5, 10, 'NA'],
            'condName': ['ts1', 'ts2', 'ts3', 'ts4', 'ss']
        })

    def test_meta_processing_steady(self):
        meta = metadata_parser.MetadataParserBranching.fix_NAs(self.meta)
        steady_idx, ts_idx = metadata_parser.MetadataParserNonbranching.process_groups(meta)
        assert len(steady_idx.keys()) == 5
        assert sum(steady_idx.values()) == 1
        assert steady_idx["ss"]

    def test_meta_processing_time(self):
        meta = metadata_parser.MetadataParserBranching.fix_NAs(self.meta)
        steady_idx, ts_idx = metadata_parser.MetadataParserNonbranching.process_groups(meta)
        assert len(ts_idx.keys()) == 4
        assert ts_idx["ts1"] == [(None, None), ("ts2", 3)]
        assert ts_idx["ts2"] == [("ts1", 3), ("ts3", 2)]
        assert ts_idx["ts3"] == [("ts2", 2), ("ts4", 5)]
        assert ts_idx["ts4"] == [("ts3", 5), (None, None)]


@pytest.mark.skip
class TestDRModelOrganisms:

    def test_on_bsubtilis(self):
        exp_data = utils.df_from_tsv('data/bsubtilis/expression.tsv')
        meta_data = utils.df_from_tsv('data/bsubtilis/meta_data.tsv', has_index=False)
        expected_design = utils.df_from_tsv('data/bsubtilis/bsubtilis_design_matrix.tsv')
        expected_response = utils.df_from_tsv('data/bsubtilis/bsubtilis_response_matrix.tsv')
        drd = design_response_translation.PythonDRDriver(tau = 45, deltmin=0, deltmax=110)
        design, response = drd.run(exp_data, meta_data)

        np.testing.assert_allclose(response.values, expected_response.values, atol=1e-15)
        assert len(set(expected_response.columns)) == len(set(response.columns))
        assert expected_response.columns.tolist() == response.columns.tolist()
        assert expected_response.index.tolist() == response.index.tolist()
        assert pd.DataFrame.equals(expected_design, design)


class TestDRAboveDeltMax:

    def setup_method(self):
        self.meta = pd.DataFrame({
            'isTs': [True, True, True, True, False],
            'is1stLast': ['f', 'm', 'm', 'l', 'e'],
            'prevCol': ['NA', 'ts1', 'ts2', 'ts3', 'NA'],
            'del.t': ['NA', 3, 2, 5, 'NA'],
            'condName': ['ts1', 'ts2', 'ts3', 'ts4', 'ss']
        })
        self.exp = pd.DataFrame(np.reshape(range(10), (2, 5)) + 1,
                                index=['gene' + str(i + 1) for i in range(2)],
                                columns=['ts' + str(i + 1) for i in range(4)] + ['ss'])

        self.drd = design_response_translation.PythonDRDriver(tau=2, deltmin=2, deltmax=4)

        self.delT_min = 2
        self.delT_max = 4
        self.tau = 2
        self.design, self.response = self.drd.run(self.exp, self.meta)

    def test_design_matrix_above_delt_max(self):
        # Set up variables
        ds, resp = (self.design, self.response)
        assert ds.shape == (2, 4)
        assert list(ds.columns) == ['ts1-ts2', 'ts2-ts3', 'ss', 'ts4'], \
            "Guarantee that the ts3-ts4 condition is dropped, since its delT of 5 is greater than delt_max of 4"
        assert list(ds['ss']) == [5, 10]
        assert list(ds['ss']) == list(resp['ss']), \
            'Steady State design and response should be equal'
        assert (resp['ts2-ts3'].values == [3, 8]).all()

    def test_response_matrix_steady_state_above_delt_max(self):
        ds, resp = (self.design, self.response)
        assert list(resp.columns) == ['ts1-ts2', 'ts2-ts3', 'ss', 'ts4']
        assert list(resp['ts4']) == list(self.exp['ts4'])
        assert list(resp['ss']) == list(self.exp['ss'])

    def test_response_matrix_time_series_above_delt_max(self):
        ds, resp = (self.design, self.response)
        expression_1 = np.array(list(self.exp['ts1']))
        expression_2 = np.array(list(self.exp['ts2']))
        expected_response_1 = (expression_1 + self.tau * (expression_2 - expression_1) / (
            float(self.meta['del.t'][1])))
        expression_3 = np.array(list(self.exp['ts3']))
        expected_response_2 = expression_2 + self.tau * (expression_3 - expression_2) / (
            float(self.meta['del.t'][2]))
        np.testing.assert_almost_equal(np.array(resp['ts1-ts2']), expected_response_1)
        np.testing.assert_almost_equal(np.array(resp['ts2-ts3']), expected_response_2)


class TestDR:
    """
    Superclass for common methods
    """

    def calculate_design_and_response(self):
        # drd = design_response_R.DRDriver()
        drd = design_response_translation.PythonDRDriver()
        target = drd.target_directory = os.path.join(my_dir, "artifacts")
        if not os.path.exists(target):
            os.makedirs(target)
        drd.delTmin = self.delT_min
        drd.delTmax = self.delT_max
        drd.tau = self.tau
        (self.design, self.response) = drd.run(self.exp, self.meta)


class TestDRMicro(TestDR):

    def setup_method(self):
        self.meta = pd.DataFrame()
        self.meta['isTs'] = [False, False]
        self.meta['is1stLast'] = ['e', 'e']
        self.meta['prevCol'] = ['NA', 'NA']
        self.meta['del.t'] = ['NA', 'NA']
        self.meta['condName'] = ['ss1', 'ss2']
        self.exp = pd.DataFrame(
            np.reshape(range(4), (2, 2)) + 1,
            index=['gene' + str(i + 1) for i in range(2)],
            columns=['ss1', 'ss2'])
        self.delT_min = 2
        self.delT_max = 4
        self.tau = 2
        self.calculate_design_and_response()

    def test_micro(self):
        ds, resp = (self.design, self.response)
        assert ds.shape == (2, 2)
        assert (ds['ss1'].values == [1, 3]).all()
        assert (ds['ss2'].values == [2, 4]).all()
        # In steady state, expect design and response to be identical
        assert ds.equals(resp)


class TestDRBelowDeltMin(TestDR):

    def setup_method(self):
        self.meta = pd.DataFrame()
        self.meta['isTs'] = [True, True, True, True, False]
        self.meta['is1stLast'] = ['f', 'm', 'm', 'l', 'e']
        self.meta['prevCol'] = ['NA', 'ts1', 'ts2', 'ts3', 'NA']
        self.meta['del.t'] = ['NA', 1, 2, 3, 'NA']
        self.meta['condName'] = ['ts1', 'ts2', 'ts3', 'ts4', 'ss']
        self.exp = pd.DataFrame(
            np.reshape(range(10), (2, 5)) + 1,
            index=['gene' + str(i + 1) for i in range(2)],
            columns=['ts' + str(i + 1) for i in range(4)] + ['ss'])
        self.delT_min = 2
        self.delT_max = 4
        self.tau = 2
        self.calculate_design_and_response()

    @pytest.mark.skip(reason="I'm not sure this is the behavior I want")
    def test_response_matrix_below_delt_min(self):
        ds, resp = (self.design, self.response)
        expression_1 = np.array(list(self.exp['ts1']))
        expression_3 = np.array(list(self.exp['ts3']))
        expected_response_1 = expression_1 + self.tau * (expression_3 - expression_1) / (
                    float(self.meta['del.t'][1]) + float(self.meta['del.t'][2]))
        np.testing.assert_almost_equal(np.array(resp['ts1-ts3']), expected_response_1)
        # pdb.set_trace()

    @pytest.mark.skip(reason="skipping until we've determined if we want to modify the legacy R code")
    def test_design_matrix_headers_below_delt_min(self):
        ds, resp = (self.design, self.response)
        print(ds.columns)
        assert list(ds.columns) == ['ss', 'ts1', 'ts2', 'ts3'], \
            "Guarantee that the ts4 condition is dropped, since its the last in the time series"


class TestBranchingTimeSeries(TestDR):

    def setup_method(self):
        self.meta = pd.DataFrame()
        self.meta['isTs'] = [True, True, True]
        self.meta['is1stLast'] = ['f', 'l', 'l']
        self.meta['prevCol'] = ['NA', 'ts1', 'ts1']
        self.meta['del.t'] = ['NA', 2, 2]
        self.meta['condName'] = ['ts1', 'ts2', 'ts3']
        self.exp = pd.DataFrame(np.reshape(range(9), (3, 3)) + 1,
                                index=['gene' + str(i + 1) for i in range(3)],
                                columns=['ts' + str(i + 1) for i in range(3)])
        self.delT_min = 1
        self.delT_max = 4
        self.tau = 1
        self.calculate_design_and_response()

    def test_design_matrix_branching_time_series(self):
        ds, resp = (self.design, self.response)
        assert ds.shape == (3, 2)
        assert list(ds.columns) == ['ts1-ts2', 'ts1-ts3'], \
            'This is how the R code happens to name branching time series'
        for col in ds:
            assert list(ds[col]) == list(self.exp['ts1']), \
                '{} column in the design matrix should be equal to the branching source, ts1, in the exp matrix'.format(col)

    def test_response_matrix_branching_time_series(self):
        ds, resp = (self.design, self.response)
        assert resp.shape == (3, 2)
        expression_1 = np.array(list(self.exp['ts1']))
        expression_2 = np.array(list(self.exp['ts2']))
        expected_response_1 = (expression_1 + self.tau * (expression_2 - expression_1) /
                               float(self.meta['del.t'][1]))

        expression_3 = np.array(list(self.exp['ts3']))
        expected_response_2 = (expression_1 + self.tau * (expression_3 - expression_1) /
                               float(self.meta['del.t'][2]))

        np.testing.assert_almost_equal(np.array(resp['ts1-ts2']), expected_response_1)
        np.testing.assert_almost_equal(np.array(resp['ts1-ts3']), expected_response_2)
