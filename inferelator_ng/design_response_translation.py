from inferelator_ng import utils
import pandas as pd
import numpy as np

DEFAULT_tau = 45
DEFAULT_delTmin = 0
DEFAULT_delTmax = 120

TS_COLUMN_NAME = 'isTs'
PREV_COLUMN_NAME = 'prevCol'
DELT_COLUMN_NAME = 'del.t'
COND_COLUMN_NAME = 'condName'


class PythonDRDriver:
    # Parameters for response matrix
    tau = DEFAULT_tau
    delTmin = DEFAULT_delTmin
    delTmax = DEFAULT_delTmax

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
    exp_data = None
    conds = None

    # Metadata
    meta_data = None
    steady_idx = None
    ts_group = None

    # Metadata column names
    ts_col = TS_COLUMN_NAME
    prev_col = PREV_COLUMN_NAME
    delt_col = DELT_COLUMN_NAME
    cond_col = COND_COLUMN_NAME

    # Output data
    included = None
    col_labels = None
    design = None
    response = None
    response_half = None

    def __init__(self, tau=DEFAULT_tau, deltmin=DEFAULT_delTmin, deltmax=DEFAULT_delTmax, return_half_tau=False):
        self.tau = tau
        self.delTmin = deltmin
        self.delTmax = deltmax
        self.return_half_tau = return_half_tau

    def run(self, exp_data, meta_data):
        """
        Process expression data and metadata into design & response data
        :param exp_data: pd.DataFrame [G x N]
        :param meta_data: pd.DataFrame [N x 5]
        :return design, response: pd.DataFrame [G x N], pd.DataFrame [G x N]
        """
        (k, n) = exp_data.shape
        self.exp_data = exp_data
        self.meta_data = meta_data

        self._fix_NAs()  # Turn NA in the dataframe into np.NaN
        self._process_groups()  # Parse metadata into dicts
        self._check_for_dupes()  # Make sure that the conditions can be properly matched with metadata

        # Pull apart the expression dataframe into indexes and an ndarray
        genes = exp_data.index.values
        self.conds = exp_data.columns.values.astype(str)
        self.exp_data = exp_data.values.astype(np.dtype('float64'))

        # Construct empty arrays for the output data
        self.col_labels = []
        self.included = np.zeros((n, 1), dtype=bool)
        self.design = np.zeros((k, 0), dtype=float)
        self.response = np.zeros((k, 0), dtype=float)
        if self.return_half_tau:
            self.response_half = np.zeros((k, 0), dtype=float)

        # Walk through all the conditions in the expression data
        for c, cc in enumerate(self.conds):
            utils.Debug.vprint("Processing condition {cc} [{c} / {tot}]".format(cc=cc, c=c + 1, tot=n), level=3)
            if self.steady_idx[cc]:
                # This is a steady-state experiment
                self.static_exp(c)
            else:
                # This is a timecourse experiment
                for prev_cond, prev_delt in self._get_prior_timepoints(cc):
                    self.timecourse_exp(c, self._get_index(prev_cond), prev_delt)
                    if not self.deep_walk_timecourse_exps:
                        break

        for c in np.where(~self.included)[0].tolist():
            # Run anything that wasn't included initially in as a steady-state experiment
            self.static_exp(c)

        self.design = pd.DataFrame(self.design, index=genes, columns=self.col_labels)
        self.response = pd.DataFrame(self.response, index=genes, columns=self.col_labels)
        if self.return_half_tau:
            self.response_half = pd.DataFrame(self.response_half, index=genes, columns=self.col_labels)

        if self.return_half_tau:
            return self.design, self.response, self.response_half
        else:
            return self.design, self.response

    def static_exp(self, idx):
        """
        Concatenate expression data onto design, response & response half
        :param idx: int
        """
        self.col_labels.append(self.conds[idx])
        self.included[idx] = True
        self.design = np.hstack((self.design, self.exp_data[:, idx].reshape(-1, 1)))
        self.response = np.hstack((self.response, self.exp_data[:, idx].reshape(-1, 1)))

        if self.return_half_tau:
            self.response_half = np.hstack((self.response_half, self.exp_data[:, idx].reshape(-1, 1)))

    def timecourse_exp(self, idx, prev_idx, prev_delt):
        """
        Concatenate expression data from the prior timepoint onto design.
        Calculate response data based on timecourse and concatenate the result onto response & response half.
        :param idx: int
        :param prev_idx: int
        :param prev_delt: numeric
        """
        new_cond = str(self.conds[prev_idx]) + "-" + str(self.conds[idx])
        self.col_labels.append(new_cond)
        self.included[idx] = True
        self.included[prev_idx] = True

        diff = self.exp_data[:, idx] - self.exp_data[:, prev_idx]
        resp = float(self.tau) / float(prev_delt) * diff + self.exp_data[:, prev_idx]

        self.design = np.hstack((self.design, self.exp_data[:, prev_idx].reshape(-1, 1)))
        self.response = np.hstack((self.response, resp.reshape(-1, 1)))

        if self.return_half_tau:
            half_resp = float(self.tau) / 2 / float(prev_delt) * diff + self.exp_data[:, prev_idx]
            self.response_half = np.hstack((self.response_half, half_resp.reshape(-1, 1)))

    def _process_groups(self):
        """
        Parse the metadata to identify steady-state experiments and link timecourse experiments
        :return steady_idx, ts_group: dict, dict
            steady_idx:  Dict keyed by condition, value is boolean (True if the condition is a steady-state experiment)
            ts_group: Dict keyed by condition. [(Previous_condition_name, Previous_delt),
                                                (Following_condition_name, Following_delt)]
        """
        time_series = self.meta_data[self.ts_col].values.astype(bool)
        steadies = dict(zip(self.meta_data[self.cond_col], np.logical_not(time_series)))

        ts_data = self.meta_data[time_series].fillna(False)
        ts_dict = dict(zip(ts_data[self.cond_col].tolist(),
                           zip(ts_data[self.prev_col].tolist(),
                               ts_data[self.delt_col].tolist())))
        ts_group = {}
        for cond, (prev, delt) in ts_dict.items():
            if prev is False or delt is False:
                prev, delt = None, None
            try:
                op, od = ts_group[cond]
                ts_group[cond] = [(prev, delt), od]
            except KeyError:
                ts_group[cond] = [(prev, delt), (None, None)]
            if prev is not None:
                try:
                    op, od = ts_group[prev]
                    ts_group[prev] = [op, (cond, delt)]
                except KeyError:
                    ts_group[prev] = [(None, None), (cond, delt)]

        self.steady_idx = steadies
        self.ts_group = ts_group

    def _check_for_dupes(self):
        """
        Sanity check the metadata and experimental data to ensure that they can be linked. Try to guess as much as
        possible unless strict checking is on
        """
        # Check to make sure that the conditions in the expression data are matched with conditions in the metadata
        exp_conds = self.exp_data.columns
        meta_conds = self.meta_data[self.cond_col]

        # Check to find out if there are any conditions which don't have associated metadata
        # If there are, raise an exception if strict_checking_for_metadata is True
        # Otherwise just assume they're steady-state data and move on
        in_exp_not_meta = exp_conds.difference(meta_conds).astype(str).tolist()
        if len(in_exp_not_meta) != 0:
            utils.Debug.vprint("{n} conditions cannot be properly matched to metadata:".format(n=len(in_exp_not_meta)),
                               level=1)
            utils.Debug.vprint(" ".join(in_exp_not_meta), level=2)
            if self.strict_checking_for_metadata:
                raise ConditionDoesNotExistError("Conditions exist without associated metadata")
            else:
                for condition in in_exp_not_meta:
                    self.steady_idx[condition] = True

        # Check to find out if the conditions in the expression data are all unique
        # It's not a problem if they're not, we just assume the metadata applies to all conditions with the same name
        duplicate_in_exp = self.exp_data.columns.duplicated()
        n_dup_in_exp = np.sum(duplicate_in_exp)
        if n_dup_in_exp > 0:
            c_dup = self.exp_data.columns[duplicate_in_exp].astype(str).tolist()
            utils.Debug.vprint("The expression data has {n} non-unique condition indexes".format(n=n_dup_in_exp),
                               level=1)
            utils.Debug.vprint(" ".join(c_dup), level=2)

        # Check to find out if the conditions in the meta data are all unique
        # If there are repeats, check and see if the associated information is identical (if it is, just ignore it)
        # If there are repeated conditions with different characteristics, the outcome may be unexpected
        # (The parser will just overwrite the first ones it comes to with the characteristics of the last one)
        duplicate_in_meta = self.meta_data[self.cond_col].duplicated()
        n_dup_in_meta = np.sum(duplicate_in_meta)
        if n_dup_in_meta > 0:
            meta_dup = self.meta_data[self.cond_col][duplicate_in_meta].astype(str).tolist()
            if np.sum(np.logical_xor(self.meta_data.duplicated(), duplicate_in_meta)) > 0:
                utils.Debug.vprint("The metadata has non-unique conditions with different characteristics:", level=0)
                utils.Debug.vprint(" ".join(meta_dup), level=0)
                if self.strict_checking_for_duplicates:
                    raise MultipleConditionsError("Identical conditions have non-identical characteristics")
            else:
                utils.Debug.vprint("The metadata contains duplicate rows:", level=1)
                utils.Debug.vprint(" ".join(meta_dup), level=2)

    def _get_prior_timepoints(self, cond):
        """
        Walk backwards through timepoints until a total del.t that falls within the acceptable window is located
        :param cond:
        :return:
        """
        total_delt = 0
        for pcond, pdelt in self._prior_timepoint_generator(cond):
            total_delt += pdelt
            if self.delTmax is not None:
                if self.delTmin <= total_delt <= self.delTmax:
                    yield pcond, total_delt
                elif total_delt > self.delTmax:
                    break
            else:
                if self.delTmin <= total_delt:
                    yield pcond, total_delt

    def _prior_timepoint_generator(self, cond):
        """
        Yield the previous timepoint condition name and delt until it gets back to the first timepoint
        :param cond:
        """
        prev_cond, prev_delt = self.ts_group[cond][0]
        while prev_cond is not None:
            yield prev_cond, prev_delt
            prev_cond, prev_delt = self.ts_group[prev_cond][0]

    def _get_index(self, cond):
        """
        Look up the index in the expression data of a specific condition. Raise errors if it doesn't exist, or if it's
        not unique
        :param cond:
        :return idx: int
        """
        idx = np.where(self.conds == cond)[0].tolist()
        if len(idx) == 0:
            raise ConditionDoesNotExistError("{cond} cannot be identified in expression conditions".format(cond=cond))
        if len(idx) > 1:
            raise MultipleConditionsError("{cond} is not unique in expression conditions".format(cond=cond))
        else:
            idx = idx[0]
        return idx

    def _fix_NAs(self):
        self.meta_data = self.meta_data.replace('NA', np.nan, regex=False)


class ConditionDoesNotExistError(IndexError):
    pass


class MultipleConditionsError(ValueError):
    pass
