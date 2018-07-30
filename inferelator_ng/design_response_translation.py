import os
from . import utils
import pandas as pd
import numpy as np

DEFAULT_tau = 45
DEFAULT_delTmin = 0
DEFAULT_delTmax = 120

TS_col_name = 'isTs'
PREV_col_name = 'prevCol'
DELTA_T_col_name = 'del.t'
CONDITION_col_name = 'condName'


class PythonDRDriver:
    # Parameters for response matrix
    tau = DEFAULT_tau
    delTmin = DEFAULT_delTmin
    delTmax = DEFAULT_delTmax
    strict_checking_for_metadata = False
    strict_checking_for_duplicates = True
    return_half_tau = False

    # Expression data
    exp_data = None

    # Metadata
    meta_data = None
    steady_idx = None
    ts_group = None

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
        (k, n) = exp_data.shape
        self.exp_data = exp_data
        self.meta_data = meta_data

        self._fix_NAs()
        self._process_groups()
        self._check_for_dupes()

        genes = exp_data.index.values
        self.conds = exp_data.columns.values
        self.exp_data = exp_data.values.astype(np.dtype('float64'))

        # Construct empty arrays for the output data
        self.col_labels = []
        self.included = np.zeros((n, 1), dtype=bool)
        self.design = np.zeros((k, 0), dtype=float)
        self.response = np.zeros((k, 0), dtype=float)
        self.response_half = np.zeros((k, 0), dtype=float)

        # Walk through all the conditions in the expression data
        for c, cc in enumerate(self.conds):
            utils.Debug.vprint("Processing condition {cc} [{c} / {tot}]".format(cc=cc, c=c, tot=n), level=3)
            if self.steady_idx[cc]:
                # This is a steady-state experiment
                self.static_exp(c)
            else:
                # This is a timecourse experiment - figure out what came before this timepoint
                prev_cond, prev_delt = self.ts_group[cc][0]
                if prev_cond is None:
                    # Don't do anything for now, this is the first timepoint
                    continue
                elif self.delTmin <= prev_delt <= self.delTmax:
                    # This is a valid timecourse experiment
                    # Construct a design & response based on the prior timepoint
                    self.timecourse_exp(c, self._get_index(prev_cond), prev_delt)
                elif prev_delt > self.delTmax:
                    # This is a timecourse experiment but the del.t is too big
                    continue
                elif prev_delt < self.delTmin:
                    # This is a timecourse experiment but the del.t is too small
                    # Try to find a prior timepoint that is acceptable
                    back_cond, back_delt = self._walk_prior_timepoints(cc)
                    if back_cond is not None:
                        self.timecourse_exp(c, self._get_index(back_cond), back_delt)
                else:
                    continue


        for c in np.where(~self.included)[0].tolist():
            # Run anything that wasn't included initially in as a steady-state experiment
            self.static_exp(c)

        self.design = pd.DataFrame(self.design, index=genes, columns=self.col_labels)
        self.response = pd.DataFrame(self.response, index=genes, columns=self.col_labels)
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
        self.response_half = np.hstack((self.response_half, self.exp_data[:, idx].reshape(-1, 1)))

    def timecourse_exp(self, idx, prev_idx, prev_delt):
        """
        Concatenate expression data onto design.
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
        half_resp = float(self.tau) / 2 / float(prev_delt) * diff + self.exp_data[:, prev_idx]

        self.design = np.hstack((self.design, self.exp_data[:, prev_idx].reshape(-1, 1)))
        self.response = np.hstack((self.response, resp.reshape(-1, 1)))
        self.response_half = np.hstack((self.response_half, half_resp.reshape(-1, 1)))

    def _process_groups(self):
        time_series = self.meta_data[TS_col_name].values.astype(bool)
        steadies = dict(zip(self.meta_data[CONDITION_col_name], np.logical_not(time_series)))

        ts_data = self.meta_data[time_series].fillna(False)
        ts_dict = dict(zip(ts_data[CONDITION_col_name].tolist(),
                           zip(ts_data[PREV_col_name].tolist(),
                               ts_data[DELTA_T_col_name].tolist())))
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
        # Check to make sure that the conditions in the expression data are matched with conditions in the metadata
        exp_conds = self.exp_data.columns
        meta_conds = self.meta_data[CONDITION_col_name]

        # Check to find out if there are any conditions which don't have associated metadata
        # If there are, raise an exception if strict_checking_for_metadata is True
        # Otherwise just assume they're steady-state data and move on
        in_exp_not_meta = exp_conds.difference(meta_conds).tolist()
        if len(in_exp_not_meta) != 0:
            utils.Debug.vprint("The following conditions cannot be properly matched to metadata:", level=1)
            utils.Debug.vprint(" ".join(in_exp_not_meta), level=1)
            if self.strict_checking_for_metadata:
                raise ValueError("Conditions exist without associated metadata")
            else:
                for condition in in_exp_not_meta:
                    self.steady_idx[condition] = True

        # Check to find out if the conditions in the expression data are all unique
        # It's not a problem if they're not, just assume the metadata applies to all conditions with the same name

        duplicate_in_exp = self.exp_data.columns.duplicated()
        if np.sum(duplicate_in_exp) > 0:
            c_dup = self.exp_data.columns[duplicate_in_exp].tolist()
            utils.Debug.vprint("The expression data has non-unique condition indexes: {}".format(" ".join(c_dup)),
                               level=1)

        # Check to find out if the conditions in the meta data are all unique
        # If there are repeats, check and see if the associated information is identical (if it is, just ignore it)
        # If there are repeated conditions with different characteristics, the outcome may be unexpected
        duplicate_in_meta = self.meta_data[CONDITION_col_name].duplicated()
        if np.sum(duplicate_in_meta) > 0:
            meta_dup = self.meta_data[CONDITION_col_name][duplicate_in_meta].tolist()
            if np.sum(np.logical_xor(self.meta_data.duplicated(), duplicate_in_meta)) > 0:
                utils.Debug.vprint("The metadata has non-unique conditions with different characteristics:", level=0)
                utils.Debug.vprint(" ".join(meta_dup), level=0)
                if self.strict_checking_for_duplicates:
                    raise ValueError("Multiple conditions exist in the metadata with different characteristics")
            else:
                utils.Debug.vprint("The metadata contains duplicate rows: {}".format(" ".join(meta_dup)), level=1)

    def _walk_prior_timepoints(self, cond):
        total_delt = 0
        for pc, pd in self._prior_timepoint_generator(cond):
            total_delt += pd
            if self.delTmin <= total_delt <= self.delTmax:
                return pc, total_delt
            elif total_delt> self.delTmax:
                return None, None
        return None, None

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

class PythonDRDriver_old:

    def __init__(self):
        pass

    def run(self, expression_mat, metadata_dataframe):

        meta_data = metadata_dataframe.copy()
        meta_data = meta_data.replace('NA', np.nan, regex=False)
        exp_mat = expression_mat.copy()

        special_char_dictionary = {'+': 'specialplus', '-': 'specialminus', '.': 'specialperiod', '/': 'specialslash',
                                   '\\': 'special_back_slash', ')': 'special_paren_backward',
                                   '(': 'special_paren_forward', ',': 'special_comma', ':': 'special_colon',
                                   ';': 'special_semicoloon', '@': 'special_at', '=': 'special_equal',
                                   '>': 'special_great', '<': 'special_less', '[': 'special_left_bracket',
                                   ']': 'special_right_bracket', "%": 'special_percent', "*": 'special_star',
                                   '&': 'special_ampersand', '^': 'special_arrow', '?': 'special_question',
                                   '!': 'special_exclamation', '#': 'special_hashtag', "{": 'special_left_curly',
                                   '}': 'special_right_curly', '~': 'special_tilde', '`': 'special_tildesib',
                                   '$': 'special_dollar', '|': 'special_vert_bar'}

        cols = exp_mat.columns.tolist()
        for ch in special_char_dictionary.keys():
            # need this edge case for passing micro test
            if len(meta_data['condName'][~meta_data['condName'].isnull()]) > 0:
                meta_data['condName'] = meta_data['condName'].str.replace(ch, special_char_dictionary[ch])
            if len(meta_data['prevCol'][~meta_data['prevCol'].isnull()]) > 0:
                meta_data['prevCol'] = meta_data['prevCol'].str.replace(ch, special_char_dictionary[ch])
            cols = [item.replace(ch, special_char_dictionary[ch]) for item in cols]
        exp_mat.columns = cols

        cond = meta_data['condName'].copy()
        prev = meta_data['prevCol'].copy()
        delt = meta_data['del.t'].copy()
        delTmin = self.delTmin
        delTmax = self.delTmax
        tau = self.tau
        prev.loc[delt > delTmax] = np.nan
        delt.loc[delt > delTmax] = np.nan
        not_in_mat = set(cond) - set(exp_mat)
        cond_dup = cond.duplicated()
        if len(not_in_mat) > 0:
            cond = cond.str.replace('[/+-]', '.')
            prev = cond.str.replace('[/+-]', '.')
            if cond_dup != cond.duplicated():
                raise ValueError(
                    'Tried to fix condition names in meta data so that they would match column names in expression matrix, but failed')

        # check if there are condition names missing in expression matrix
        not_in_mat = set(cond) - set(exp_mat)
        if len(not_in_mat) > 0:
            print(not_in_mat)
            raise ValueError(
                'Error when creating design and response. The conditions printed above are in the meta data, but not in the expression matrix')

        cond_n_na = cond[~cond.isnull()]
        steady = prev.isnull() & ~(cond_n_na.isin(prev.replace(np.nan, "NA")))

        des_mat = pd.DataFrame(exp_mat[cond[steady]])
        res_mat = pd.DataFrame(exp_mat[cond[steady]])

        for i in list(np.where(~steady)[0]):
            following = list(np.where(prev.str.contains(cond[i]) == True)[0])
            following_delt = list(delt[following])

            try:
                off = list(np.where(following_delt[0] < delTmin)[0])
            except:
                off = []

            while len(off) > 0:
                off_fol = list(np.where(prev.str.contains(cond[following[off[0]]]) == True)[0])
                off_fol_delt = list(delt[off_fol])
                following = following[:off[0]] + following[off[0] + 1:] + off_fol
                following_delt = following_delt[:off[0]] + following_delt[off[0] + 1:] + [
                    float(off_fol_delt[0]) + float(following_delt[off[0]])]
                off = list(np.where(following_delt < [delTmin])[0])

            n = len(following)
            cntr = 0

            for j in following:
                if n > 1:
                    this_cond = "%s_dupl%02d" % (cond[i], cntr + 1)
                    original_this_cond = this_cond
                    k = 1
                    while this_cond in res_mat.columns:
                        this_cond = original_this_cond + '.{}'.format(int(k))
                        k = k + 1
                else:
                    this_cond = cond[i]

                des_tmp = np.concatenate((des_mat.values, exp_mat[cond[i]].values[:, np.newaxis]), axis=1)
                des_names = list(des_mat.columns) + [this_cond]
                des_mat = pd.DataFrame(des_tmp, index=des_mat.index, columns=des_names)
                interp_res = (float(tau) / float(following_delt[cntr])) * (
                            exp_mat[cond[j]].astype('float64') - exp_mat[cond[i]].astype('float64')) + exp_mat[
                                 cond[i]].astype('float64')
                res_tmp = np.concatenate((res_mat.values, interp_res.values[:, np.newaxis]), axis=1)
                res_names = list(res_mat.columns) + [this_cond]
                res_mat = pd.DataFrame(res_tmp, index=res_mat.index, columns=res_names)

                cntr = cntr + 1

            # special case: nothing is following this condition within delT.min
            # and it is the first of a time series --- treat as steady state

            if n == 0 and prev.isnull()[i]:
                des_mat = pd.concat([des_mat, exp_mat[cond[i]]], axis=1)
                des_mat.rename(columns={des_mat.columns.values[len(des_mat.columns) - 1]: cond[i]}, inplace=True)
                res_mat = pd.concat([res_mat, exp_mat[cond[i]]], axis=1)
                res_mat.rename(columns={res_mat.columns.values[len(res_mat.columns) - 1]: cond[i]}, inplace=True)

        cols_des_mat = des_mat.columns.tolist()
        cols_res_mat = res_mat.columns.tolist()

        special_char_inv_map = {v: k for k, v in list(special_char_dictionary.items())}
        for sch in special_char_inv_map.keys():
            cols_des_mat = [item.replace(sch, special_char_inv_map[sch]) for item in cols_des_mat]
            cols_res_mat = [item.replace(sch, special_char_inv_map[sch]) for item in cols_res_mat]

        des_mat.columns = cols_des_mat
        res_mat.columns = cols_res_mat

        return (des_mat, res_mat)
