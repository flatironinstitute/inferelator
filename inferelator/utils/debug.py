import os
import logging
import sys

SBATCH_VARS = dict(
    output_dir=('RUNDIR', str, None),
    input_dir=('DATADIR', str, None)
)


class Debug:
    """
    Wrapper for logging
    Exists for historic reasons
    Probably wouldn't do it this way from scratch
    But im not rewriting everything that depends on this
    """
    verbose_level = 0
    default_level = 1

    stderr = False
    logger = None

    levels = dict(
        silent=-1,
        normal=0,
        verbose=1, v=1,
        very_verbose=2, vv=2,
        max_output=3, vvv=3
    )

    @classmethod
    def set_verbose_level(
        cls,
        lvl
    ):
        if isinstance(lvl, (int, float)):
            cls.verbose_level = lvl
        elif lvl in cls.levels.keys():
            cls.verbose_level = cls.levels[lvl]

    @classmethod
    def vprint(
        cls,
        *args,
        **kwargs
    ):
        cls.print_level(*args, **kwargs)

    @classmethod
    def print_level(
        cls,
        *args,
        **kwargs
    ):
        level = kwargs.pop('level', cls.default_level)

        if level <= cls.verbose_level:
            cls.create_logger()

            cls.logger.log(
                level + 1,
                *args,
                **kwargs
            )

    @classmethod
    def log_to_stderr(
        cls,
        stderr_flag
    ):
        cls.stderr = stderr_flag

    @classmethod
    def create_logger(cls):

        if cls.logger is not None:
            return

        cls.logger = logging.Logger('inferelator')

        logger_handler = logging.StreamHandler(
            sys.stderr if cls.stderr else sys.stdout
        )

        logger_handler.setFormatter(
            logging.Formatter(
                '%(asctime)-15s %(levelno)s %(message)s',
                '%Y-%m-%d %H:%M:%S'
            )
        )

        cls.logger.addHandler(logger_handler)


def inferelator_verbose_level(
    level,
    log_to_stderr=None
):
    """
    Set verbosity.
    :param level: Verbose level.
        0 is normal, 1 is extra information.
        2+ is not recommended.
        -1 silences most outputs.
    :type level: int
    :param log_to_stderr: Log to stderr instead of stdout
    :type log_to_stderr: bool
    """

    Debug.set_verbose_level(level)

    if log_to_stderr is not None:
        Debug.log_to_stderr(log_to_stderr)


def slurm_envs(
    var_names=None
):
    """
    Get environment variable names and return them as a dict

    :param var_names: A list of environment variable names to get.
        Will throw an error if they're not keys in the SBATCH_VARS dict
    :type var_names: list
    :return envs: A dict keyed by setattr variable name of the value
        (or default) from the environment variables
    :rtype dict
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
