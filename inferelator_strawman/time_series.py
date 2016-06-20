import pandas as pd

class ResponseParameters:

    """Container for time series data for a gene in a condition
    Which may be used to determine the response value
    for that gene in the condition.
    """

    def __init__(self, gene_name, condition,
        gene_level_before, gene_level, time_interval):
        self.gene_name = gene_name
        self.condition = condition
        # level for gene in previous condition (or None if first)
        self.gene_level_before = gene_level_before
        # level for gene in this condition
        self.gene_level = gene_level
        self.time_interval = time_interval

class TransitionResponse:

    """
    The "y" response value for a gene in a condition in a time series.
    """
    # XXXX This is my reading of Greenfield et al. p1062.
    # -- it doesn't seem to correspond exactly to calculations in
    # https://github.com/ChristophH/Inferelator/blob/master/R_scripts/design_and_response.R
    # Therefore the current code is probably wrong.

    def __init__(self, tau_half_life):
        assert tau_half_life > 0, 'half life must be positive'
        # XXXX other restrictions on tau?
        self.tau_half_life = tau_half_life

    def gene_response(self, parameters):
        gene_level = 1.0 * parameters.gene_level
        gene_level_before = parameters.gene_level_before
        if gene_level_before is None:
            # first condition is assumed to be in a steady state.
            return gene_level
        # otherwise compute finite difference
        tau = self.tau_half_life
        interval = parameters.time_interval
        assert interval > 0, "time interval must be positive."
        level_change = 1.0 * (gene_level - gene_level_before)
        # XXXX is this right?
        result = (level_change/interval) + (gene_level / tau)
        return result

class TimeSeries:

    """
    A time series is a sequence of conditions separated by time intervals.
    """

    def __init__(self, first_condition):
        first_condition_name = first_condition.name
        self.first_condition_name = first_condition_name
        self._condition_name_to_previous_condition_name = {}
        self._conditions_by_name = {first_condition_name: first_condition}
        self._time_interval_before_condition = {}
        self._condition_name_order = None
        self._following_conditions = None

    def following_conditions(self):
        """
        Return a dictionary mapping condition name to the set of conditions that follow.
        """
        if self._following_conditions is not None:
            return self._following_conditions
        result = {}
        for name in self._conditions_by_name:
            result[name] = set()
        for (next_cond, prev_cond) in self._condition_name_to_previous_condition_name.items():
            result[prev_cond].add(next_cond)
        self._following_conditions = result
        return result

    def meta_data_tsv_lines(self):
        "Return text containing lines for self in tsv format terminated by newline"
        # XXXX This needs fixing: time series can have branching!
        L = []
        intervals_before = self._time_interval_before_condition
        follows = self.following_conditions()
        for condition in self.get_condition_order():
            conditionName = condition.name
            delt = intervals_before.get(conditionName)
            is1stLast = "f"
            follow = follows[conditionName]
            prevCol = self._condition_name_to_previous_condition_name.get(conditionName)
            if prevCol:
                is1stLast = "m"
                if not follow:
                    is1stLast = "l"
            tsvline = condition.meta_data_tsv_line(
                isTs=True,
                is1stLast=is1stLast,
                prevCol=prevCol,
                delt=delt)
            L.append(tsvline)
        return "".join(L)

    def get_response_parameters(self, condition_name, gene_name):
        # XXXX This needs fixing: time series can have branching!
        names = self.get_condition_name_order()
        intervals = self.get_interval_order()
        condition = self._conditions_by_name[condition_name]
        index = names.index(condition_name)
        interval = intervals[index]
        gene_level = condition.response_scalar(gene_name)
        gene_level_before = None
        if index > 0:
            condition_before = self._conditions_by_name[names[index-1]]
            gene_level_before = condition_before.response_scalar(gene_name)
        return ResponseParameters(gene_name, condition_name,
            gene_level_before, gene_level, interval)
        
    def get_condition_name_order(self, force=False):
        # XXXX This needs fixing: time series can have branching!
        if not force and self._condition_name_order is not None:
            return self._condition_name_order
        name_order = []
        follows = self.following_conditions()
        interval_before = self._time_interval_before_condition
        stack = [self.first_condition_name]
        conditions_seen = set()
        while stack:
            this_condition = stack.pop()
            conditions_seen.add(this_condition)
            name_order.append(this_condition)
            next_conditions = follows[this_condition]
            assert len(next_conditions & conditions_seen) == 0, (
                "Cycle found in timeseries condition description. " + repr((
                    name_order, this_condition, next_conditions)))
            stack.extend(reversed(sorted(next_conditions)))
        assert set(self._conditions_by_name) == conditions_seen, (
            "not all conditions ordered:" + 
            repr((set(self._conditions_by_name), conditions_seen)))
        self._condition_name_order = name_order
        return name_order

    def get_condition_order(self):
        cbn = self._conditions_by_name
        name_order = self.get_condition_name_order()
        return [cbn[name] for name in name_order]
        
    def get_interval_order(self):
        name_order = self.get_condition_name_order()
        before = self._time_interval_before_condition
        # Assume "interval before first" is zero
        return [before.get(name, 0) for name in name_order]
        
    def add_condition(self, prev_condition_name, condition, time_interval_before_condition):
        # XXXX This needs fixing: time series can have branching!
        assert self._condition_name_order is None, (
            "Cannot modify time series after it has been compiled into an ordered sequence."
        )
        name = condition.name
        assert name not in self._conditions_by_name, (
            "duplicate condition: " + repr(name)
        )
        self._conditions_by_name[name] = condition
        self._time_interval_before_condition[name] = time_interval_before_condition
        self._condition_name_to_previous_condition_name[name] = prev_condition_name
