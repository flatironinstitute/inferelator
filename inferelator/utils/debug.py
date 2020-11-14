from __future__ import print_function, unicode_literals, division
import os

from inferelator.default import SBATCH_VARS


class Debug:
    """
    This class is for printing status messages to stdout
    Just plain print doesn't work so well when there are multiple processes
    """
    verbose_level = 0
    default_level = 1

    silence_clients = True
    is_master = True

    levels = dict(silent=-1,
                  normal=0,
                  verbose=1, v=1,
                  very_verbose=2, vv=2,
                  max_output=3, vvv=3)

    @classmethod
    def set_verbose_level(cls, lvl):
        if isinstance(lvl, (int, float)):
            cls.verbose_level = lvl
        elif lvl in cls.levels.keys():
            cls.verbose_level = cls.levels[lvl]

    @classmethod
    def vprint(cls, *args, **kwargs):
        if cls.silence_clients and not cls.is_master:
            return
        cls.print_level(*args, **kwargs)

    @classmethod
    def allprint(cls, *args, **kwargs):
        cls.print_level(*args, **kwargs)

    @classmethod
    def print_level(cls, *args, **kwargs):
        try:
            level = kwargs.pop('level')
        except KeyError:
            level = cls.default_level
        if level <= cls.verbose_level:
            print((" " * level), *args, **kwargs)
        else:
            return


def inferelator_verbose_level(level):
    """
    Set verbosity.
    :param level: Verbose level. 0 is normal, 1 is extra information. 2+ is not recommended. -1 silences most outputs.
    :type level: int
    """
    Debug.set_verbose_level(level)


def slurm_envs(var_names=None):
    """
    Get environment variable names and return them as a dict
    :param var_names: list
        A list of environment variable names to get. Will throw an error if they're not keys in the SBATCH_VARS dict
    :return envs: dict
        A dict keyed by setattr variable name of the value (or default) from the environment variables
    """
    var_names = SBATCH_VARS.keys() if var_names is None else var_names
    assert set(var_names).issubset(set(SBATCH_VARS.keys()))

    envs = {}
    for cv in var_names:
        os_var, mt, de = SBATCH_VARS[cv]
        try:
            val = mt(os.environ[os_var])
        except (KeyError, TypeError):
            val = de
        envs[cv] = val
    return envs