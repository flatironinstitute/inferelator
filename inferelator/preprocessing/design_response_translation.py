from __future__ import division

from inferelator import utils
from inferelator import default
from inferelator.preprocessing.metadata_parser import MetadataHandler
from inferelator.preprocessing.metadata_parser import ConditionDoesNotExistError, MultipleConditionsError
import pandas as pd
import numpy as np


class PythonDRDriver(object):
    # Parameters for response matrix
    tau = default.DEFAULT_TAU
    delTmin = default.DEFAULT_DELTMIN
    delTmax = default.DEFAULT_DELTMAX

    # Strict checking will raise exception if there are potential inconsistencies
    # Strict_checking_for_metadata raises an exception if there are any missing experiments in the metadata
    # Strict_checking_for_duplicates raises an exception if there are non-trivial duplicates in the metadata
    strict_checking_for_metadata = False
    strict_checking_for_duplicates = True

    # Deep walking timecourse experiments means generating separate conditions for every linked experiment within the
    # acceptable time window. If set to false, it will just use the most recent linked experiment.
    deep_walk_timecourse_exps = False

    # Set return_half_tau to true to return the half_tau_response_matrix
    return_half_tau = False

    # Expression data
    sample_names = None

    # Metadata
    steady_idx = None
    ts_group = None

    def __init__(self, tau=None, deltmin=None, deltmax=None, return_half_tau=False):
        self.tau = tau if tau is not None else default.DEFAULT_TAU
        self.delTmin = deltmin if deltmin is not None else default.DEFAULT_DELTMIN
        self.delTmax = deltmax if deltmax is not None else default.DEFAULT_DELTMAX
        self.return_half_tau = return_half_tau

    def run(self, exp_data, meta_data):
        """
        Process expression data and metadata into design & response data
        :param exp_data: pd.DataFrame [G x N]
        :param meta_data: pd.DataFrame [N x 5]
        :return design, response: pd.DataFrame [G x N], pd.DataFrame [G x N]
        """

        (k, n) = exp_data.shape
        processor = MetadataHandler.get_handler()

        # Turn NA in the dataframe into np.NaN
        meta_data = processor.fix_NAs(meta_data)

        # Validate metadata alignment to expression
        processor.validate_metadata(exp_data, meta_data)

        # Turn the metadata into a set of dicts keyed by sample
        steady_idx, ts_group = processor.process_groups(meta_data)

        # Check and make sure that the metadata matches experimental data and whatnot
        steady_idx = processor.check_for_dupes(exp_data, meta_data, steady_idx,
                                               strict_checking_for_metadata=self.strict_checking_for_metadata,
                                               strict_checking_for_duplicates=self.strict_checking_for_duplicates)

        # Pull apart the expression dataframe into indexes and an ndarray
        genes = exp_data.index.values
        self.sample_names = exp_data.columns.values.astype(str)
        exp_data = exp_data.values.astype(np.dtype('float64'))

        # Construct empty arrays for the output data
        col_labels = []
        included = np.zeros((n, 1), dtype=bool)
        design = []
        response = []
        response_half = []

        # Walk through all the conditions in the expression data
        for c_idx, cc in enumerate(self.sample_names):
            utils.Debug.vprint("Processing condition {cc} [{c} / {tot}]".format(cc=cc, c=c_idx + 1, tot=n), level=3)
            if steady_idx[cc]:
                # This is a steady-state experiment
                self.static_exp(c_idx, cc, col_labels, included, exp_data, design, response, response_half)
            else:
                # This is a timecourse experiment
                for prev_cond, prev_delt in self._get_prior_timepoints(ts_group, cc):
                    prev_idx = self._get_index(self.sample_names, prev_cond)
                    self.timecourse_exp(c_idx, prev_idx, prev_delt, col_labels, included, exp_data, design, response,
                                        response_half)
                    if not self.deep_walk_timecourse_exps:
                        break

        for c_idx in np.where(~included)[0].tolist():
            # Run anything that wasn't included initially in as a steady-state experiment
            cc = self.sample_names[c_idx]
            self.static_exp(c_idx, cc, col_labels, included, exp_data, design, response, response_half)

        design = pd.DataFrame(np.array(design), index=col_labels, columns=genes).transpose()
        response = pd.DataFrame(np.array(response), index=col_labels, columns=genes).transpose()

        if self.return_half_tau:
            response_half = pd.DataFrame(np.array(response_half), index=col_labels, columns=genes).transpose()
            return design, response, response_half
        else:
            return design, response

    @staticmethod
    def static_exp(c_idx, c_name, col_labels, included, exp_data, design, response, response_half):
        """
        Concatenate expression data onto design, response & response half

        :param c_idx: int
        :param c_name: str
        :param col_labels: list
        :param included: dict
        :param exp_data: np.ndarray
        :param design: list(np.ndarray)
        :param response: list(np.ndarray)
        :param response_half: list(np.ndarray)
        """
        col_labels.append(c_name)
        included[c_idx] = True
        design.append(exp_data[:, c_idx].flatten())
        response.append(exp_data[:, c_idx].flatten())
        response_half.append(exp_data[:, c_idx].flatten())

    def timecourse_exp(self, c_idx, prev_idx, prev_delt, col_labels, included, exp_data, design, response,
                       response_half):
        """
        Concatenate expression data from the prior timepoint onto design.
        Calculate response data based on timecourse and concatenate the result onto response & response half.
        :param c_idx: int
        :param prev_idx: int
        :param prev_delt: numeric
        :param col_labels: list
        :param included: dict
        :param exp_data: np.ndarray
        :param design: list(np.ndarray)
        :param response: list(np.ndarray)
        :param response_half: list(np.ndarray)
        """

        col_labels.append(str(self.sample_names[prev_idx]) + "-" + str(self.sample_names[c_idx]))
        included[c_idx] = True
        included[prev_idx] = True

        resp, half_resp = self._calculate_ts_response(exp_data, self.tau, prev_delt, c_idx, prev_idx)

        design.append(exp_data[:, prev_idx].flatten())
        response.append(resp.flatten())
        response_half.append(half_resp.flatten())

    def _get_prior_timepoints(self, ts_group, cond):
        """
        Walk backwards through timepoints until a total del.t that falls within the acceptable window is located
        :param cond:
        :return:
        """
        total_delt = 0
        for pcond, pdelt in self._prior_timepoint_generator(ts_group, cond):
            total_delt += pdelt
            if self.delTmax is not None:
                if self.delTmin <= total_delt <= self.delTmax:
                    yield pcond, total_delt
                elif total_delt > self.delTmax:
                    break
            else:
                if self.delTmin <= total_delt:
                    yield pcond, total_delt

    @staticmethod
    def _calculate_ts_response(exp_data, tau, prev_delt, idx, prev_idx):
        diff = exp_data[:, idx] - exp_data[:, prev_idx]
        resp = float(tau) / float(prev_delt) * diff + exp_data[:, prev_idx]
        half_resp = float(tau) / 2 / float(prev_delt) * diff + exp_data[:, prev_idx]
        return resp, half_resp

    @staticmethod
    def _prior_timepoint_generator(ts_group, cond):
        """
        Yield the previous timepoint condition name and delt until it gets back to the first timepoint
        :param cond:
        """
        prev_cond, prev_delt = ts_group[cond][0]
        while prev_cond is not None:
            yield prev_cond, prev_delt
            prev_cond, prev_delt = ts_group[prev_cond][0]

    @staticmethod
    def _get_index(sample_names, cond):
        """
        Look up the index in the expression data of a specific condition. Raise errors if it doesn't exist, or if it's
        not unique
        :param sample_names: list
        :param cond: str
        :return idx: int
        """
        idx = np.where(sample_names == cond)[0].tolist()
        if len(idx) == 0:
            raise ConditionDoesNotExistError("{cond} cannot be identified in expression conditions".format(cond=cond))
        if len(idx) > 1:
            raise MultipleConditionsError("{cond} is not unique in expression conditions".format(cond=cond))
        else:
            idx = idx[0]
        return idx
