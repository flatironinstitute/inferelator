import numpy as np
import pandas as pd
from abc import abstractmethod

from inferelator import utils

ISTS_COLUMN_NAME = 'isTs'
PREV_COLUMN_NAME = 'prevCol'
DELT_COLUMN_NAME = 'del.t'
COND_COLUMN_NAME = 'condName'

GROUP_COLUMN_NAME = "strain"
TIME_COLUMN_NAME = "time"

DEFAULT_STRICT_CHECKING_FOR_METADATA = False
DEFAULT_STRICT_CHECKING_FOR_DUPLICATES = True


class MetadataParser(object):

    @classmethod
    @abstractmethod
    def process_groups(cls, meta_data):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def check_for_dupes(cls, exp_data, meta_data, steady_idx,
                        strict_checking_for_metadata=DEFAULT_STRICT_CHECKING_FOR_METADATA,
                        strict_checking_for_duplicates=DEFAULT_STRICT_CHECKING_FOR_DUPLICATES):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def validate_metadata(cls, exp_data, meta_data):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_default_meta_data(cls, expression_matrix):
        raise NotImplementedError

    @staticmethod
    def fix_NAs(data_frame):
        """
        Replace the string NA with a np.nan
        :param data_frame: pd.DataFrame
        :return: pd.DataFrame
        """
        return data_frame.replace('NA', np.nan, regex=False)


class MetadataParserBranching(MetadataParser):
    """
    Metadata parser that handles prev_col & del_t metadata (which allows for Branching Time Courses)
    """

    ists_col = ISTS_COLUMN_NAME  # Column of booleans for "Is a time-series experiment"
    cond_col = COND_COLUMN_NAME  # Column of sample names (matching expression data column names)
    prev_col = PREV_COLUMN_NAME  # Column that identifies the previous timepoint sample name
    delt_col = DELT_COLUMN_NAME  # Column that identifies the delta time between this sample and the previous

    @classmethod
    def process_groups(cls, meta_data):
        """
        Parse the metadata to identify steady-state experiments and link timecourse experiments
        :param meta_data: pd.DataFrame
        :return steady_idx, ts_group: dict, dict
            steady_idx:  Dict keyed by condition, value is boolean (True if the condition is a steady-state experiment)
            ts_group: Dict keyed by condition. [(Previous_condition_name, Previous_delt),
                                                (Following_condition_name, Following_delt)]
        """
        time_series = meta_data[cls.ists_col].values.astype(bool)
        steadies = dict(zip(meta_data[cls.cond_col].astype(str), np.logical_not(time_series)))

        ts_data = meta_data[time_series].fillna(False)
        ts_dict = dict(zip(ts_data[cls.cond_col].astype(str).tolist(),
                           zip(ts_data[cls.prev_col].tolist(),
                               ts_data[cls.delt_col].tolist())))
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

        return steadies, ts_group

    @classmethod
    def check_for_dupes(cls, exp_data, meta_data, steady_idx,
                        strict_checking_for_metadata=DEFAULT_STRICT_CHECKING_FOR_METADATA,
                        strict_checking_for_duplicates=DEFAULT_STRICT_CHECKING_FOR_DUPLICATES):
        """
        Sanity check the metadata and experimental data to ensure that they can be linked. Try to guess as much as
        possible unless strict checking is on
        """
        # Check to make sure that the conditions in the expression data are matched with conditions in the metadata
        exp_conds = exp_data.columns.astype(str)
        meta_conds = meta_data[cls.cond_col].astype(str)

        # Check to find out if there are any conditions which don't have associated metadata
        # If there are, raise an exception if strict_checking_for_metadata is True
        # Otherwise just assume they're steady-state data and move on
        in_exp_not_meta = exp_conds.difference(meta_conds).tolist()
        if len(in_exp_not_meta) != 0:
            utils.Debug.vprint("{n} conditions cannot be properly matched to metadata".format(n=len(in_exp_not_meta)),
                               level=1)
            utils.Debug.vprint(" ".join(in_exp_not_meta), level=2)
            if strict_checking_for_metadata:
                raise ConditionDoesNotExistError("Conditions exist without associated metadata")
            else:
                for condition in in_exp_not_meta:
                    steady_idx[condition] = True

        # Check to find out if the conditions in the expression data are all unique
        # It's not a problem if they're not, we just assume the metadata applies to all conditions with the same name
        duplicate_in_exp = exp_conds.duplicated()
        n_dup_in_exp = np.sum(duplicate_in_exp)
        if n_dup_in_exp > 0:
            c_dup = exp_conds[duplicate_in_exp].tolist()
            utils.Debug.vprint("The expression data has {n} non-unique condition indexes".format(n=n_dup_in_exp),
                               level=1)
            utils.Debug.vprint(" ".join(c_dup), level=2)

        # Check to find out if the conditions in the meta data are all unique
        # If there are repeats, check and see if the associated information is identical (if it is, just ignore it)
        # If there are repeated conditions with different characteristics, the outcome may be unexpected
        # (The parser will just overwrite the first ones it comes to with the characteristics of the last one)
        duplicate_in_meta = meta_conds.duplicated()
        n_dup_in_meta = np.sum(duplicate_in_meta)
        if n_dup_in_meta > 0:
            meta_dup = meta_conds[duplicate_in_meta].tolist()
            if np.sum(np.logical_xor(meta_data.duplicated(), duplicate_in_meta)) > 0:
                utils.Debug.vprint("The metadata has non-unique conditions with different characteristics:", level=0)
                utils.Debug.vprint(" ".join(meta_dup), level=0)
                if strict_checking_for_duplicates:
                    raise MultipleConditionsError("Identical conditions have non-identical characteristics")
            else:
                utils.Debug.vprint("The metadata contains {n} duplicate rows".format(n=n_dup_in_meta), level=1)
                utils.Debug.vprint(" ".join(meta_dup), level=2)

        return steady_idx

    @classmethod
    def validate_metadata(cls, exp_data, meta_data):
        """
        Make sure that meta_data and expression_data are compatible
        """

        # Check and make sure that there's a name column and create it if needed
        cls.create_sample_name_column(meta_data)

        # Check the alignment of the expression data and the meta_data
        sample_names_expr = exp_data.columns.astype(str)
        sample_names_meta = meta_data[cls.cond_col].astype(str)
        align_count = len(sample_names_expr.intersection(sample_names_meta))

        if align_count == 0:
            raise ConditionDoesNotExistError("Unable to align metadata to expression data")
        elif align_count < min(exp_data.shape[1], meta_data.shape[0]):
            utils.Debug.vprint("Metadata ({me}) and expression data ({ex}) alignment off".format(me=meta_data.shape,
                                                                                                 ex=exp_data.shape),
                               level=0)

    @classmethod
    def create_sample_name_column(cls, meta_data):
        """
        Create a meta_data sample_name column from the index if necessary
        """
        if cls.cond_col not in meta_data:
            meta_data[cls.cond_col] = meta_data.index.astype(str)
        elif (meta_data[cls.cond_col] != meta_data.index.astype(str)).any():
            utils.Debug.vprint("Meta data sample name column and meta_data index are not equal", level=2)

    @classmethod
    def create_default_meta_data(cls, expression_matrix):
        """
        Create a meta_data dataframe from basic defaults
        """

        # Create an empty dataframe with index equal to sample names from expression data
        meta_data = pd.DataFrame(index=expression_matrix.columns.astype(str))

        # Create a name column
        cls.create_sample_name_column(meta_data)

        # Assign default values to information columns
        meta_data[cls.ists_col] = True
        meta_data[cls.prev_col] = "NA"
        meta_data[cls.delt_col] = "NA"
        return meta_data


class MetadataParserNonbranching(MetadataParserBranching):
    group_col = GROUP_COLUMN_NAME
    time_col = TIME_COLUMN_NAME
    cond_col = COND_COLUMN_NAME

    default_values = {"isTs": "FALSE", "is1stLast": "e", "prevCol": "NA", "del.t": "NA", "condName": None}

    @classmethod
    def process_groups(cls, meta_data):
        ts_dict = dict(zip(meta_data[cls.cond_col].astype(str).tolist(),
                           zip(meta_data[cls.group_col].tolist(),
                               meta_data[cls.time_col].tolist())))

        times_per_group = {k: [] for k in meta_data[cls.group_col].unique()}
        group_time_cond = {k: {} for k in meta_data[cls.group_col].unique()}
        ts_group = {cond: [(None, None), (None, None)] for cond in ts_dict.keys()}
        steady_idx = {cond: False for cond in ts_dict.keys()}

        # Walk through metadata to find the experiment times for each strain
        # And the sample IDs associated with each strain/time combo
        for cond, (group, time) in ts_dict.items():
            times_per_group[group].append(time)
            group_time_cond[group][time] = cond
            if np.isnan(time):
                steady_idx[cond] = True

        # Process into a dict, keyed by sample ID, of [(previous sample, del.t), (next sample, del.t)]
        for cond, (group, time) in ts_dict.items():
            group_times = times_per_group[group]

            if (min(group_times) == time and max(group_times) == time) or steady_idx[cond]:
                steady_idx[cond] = True
                ts_group.pop(cond)
                continue

            if min(group_times) == time:
                prev = (None, None)
            else:
                prev_time = max([i for (i, v) in zip(group_times, map(lambda x: x < time, group_times)) if v])
                prev_cond = group_time_cond[group][prev_time]
                prev = (prev_cond, time - prev_time)

            if max(group_times) == time:
                nex = (None, None)
            else:
                nex_time = min([i for (i, v) in zip(group_times, map(lambda x: x > time, group_times)) if v])
                nex_cond = group_time_cond[group][nex_time]
                nex = (nex_cond, nex_time - time)

            ts_group[cond] = [prev, nex]

        return steady_idx, ts_group

    @classmethod
    def create_default_meta_data(cls, expression_matrix):
        """
        Create a meta_data dataframe from basic defaults
        """
        meta_data = pd.DataFrame(index=expression_matrix.columns)
        meta_data[cls.cond_col] = expression_matrix.columns
        meta_data[cls.group_col] = list(range(expression_matrix.shape[1]))
        meta_data[cls.time_col] = 0
        return meta_data


class MetadataHandler(object):
    """
    This keeps track of how to process metadata
    """
    handler = MetadataParserBranching

    @classmethod
    def set_handler(cls, handler_ref):
        if utils.is_string(handler_ref):
            if handler_ref == "branching":
                cls.handler = MetadataParserBranching
            elif handler_ref == "nonbranching":
                cls.handler = MetadataParserNonbranching
            else:
                raise ValueError("Parser {parser_str} unknown".format(parser_str=handler_ref))
        elif issubclass(handler_ref, MetadataParser):
            cls.handler = handler_ref
        else:
            raise ValueError("Handler must be a string or a MetadataParser class")

    @classmethod
    def get_handler(cls):
        return cls.handler

    @classmethod
    def make_default_metadata(cls, expression_data):
        return cls.handler.create_default_meta_data(expression_data)


class ConditionDoesNotExistError(IndexError):
    pass


class MultipleConditionsError(ValueError):
    pass
