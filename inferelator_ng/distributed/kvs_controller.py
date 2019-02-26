"""
KVSController is a wrapper for KVSClient that adds some useful functionality related to interprocess
communication.
It also keeps track of a bunch of SLURM related stuff that was previously workflow's problem.
"""

from kvsstcp import KVSClient

import os
import warnings

# SLURM environment variables
SBATCH_VARS = dict(SLURM_PROCID=('rank', int, 0),
                   SLURM_NTASKS_PER_NODE=('cores', int, 1),
                   SLURM_NTASKS=('tasks', int, 1),
                   SLURM_NODEID=('node', int, 0),
                   SLURM_JOB_NUM_NODES=('num_nodes', int, 1))

DEFAULT_MASTER = 0
DEFAULT_WARNING = "SBATCH has not set ENV {var}. Setting {var} to {defa}."


class KVSController:
    # Set from SLURM environment variables

    kvs_client = None

    rank = None  # int
    tasks = None  # int
    node = None  # int
    cores = None  # int
    num_nodes = None  # int
    is_master = False  # bool

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Create a new KVS object with some object variables set to reflect the slurm environment
        """

        # Get local environment variables
        cls._get_env(suppress_warnings=kwargs.pop("suppress_warnings", False),
                     master_rank=kwargs.pop("master_rank", 0))

        # Connect to the host server by calling to KVSClient.__init__
        cls.kvs_client = KVSClient(*args, **kwargs)

    @classmethod
    def _get_env(cls, slurm_variables=SBATCH_VARS, suppress_warnings=False, master_rank=DEFAULT_MASTER):
        """
        Get the SLURM environment variables that are set by sbatch at runtime.
        The default values mean multiprocessing won't work at all.
        """
        for env_var, (class_var, func, default) in slurm_variables.items():
            try:
                val = func(os.environ[env_var])
            except (KeyError, TypeError):
                val = default
                if not suppress_warnings:
                    print(DEFAULT_WARNING.format(var=env_var, defa=default))
            setattr(cls, class_var, val)
        if cls.rank == master_rank:
            cls.is_master = True
        else:
            cls.is_master = False

    @classmethod
    def own_check(cls, chunk=1, kvs_key='count'):
        if cls.is_master:
            return ownCheck(cls.kvs_client, 0, chunk=chunk, kvs_key=kvs_key)
        else:
            return ownCheck(cls.kvs_client, 1, chunk=chunk, kvs_key=kvs_key)

    @classmethod
    def master_remove_key(cls, kvs_key='count'):
        if cls.is_master:
            cls.kvs_client.get(kvs_key)

    @classmethod
    def sync_processes(cls, pref="", value=True):
        """
        Block all processes until they reach this point, then release them
        It may be wise to use unique prefixes if this is gonna get called rapidly so there's no collision
        Or not. I'm a comment, not a cop.
        :param pref: str
            Prefix attached to the KVS keys
        :param value: Anything you can pickle
            A value that will be checked for consistency between processes (if you set a different value in a
            process, a warning will be issued. This is mostly to check state if needed
        :return None:
        """


        wkey = pref + '_wait'
        ckey = pref + '_continue'

        # Every process puts a wait key up when it gets here
        cls.put_key(wkey, value)

        # The master pulls down the wait keys until it has all of them
        # Then it puts up a go key for each process

        if cls.is_master:
            for _ in range(cls.tasks):
                c_value = cls.get_key(wkey)
                if c_value != value:
                    warnings.warn("Sync warning: master {val_m} is not equal to client {val_c}".format(val_m=value,
                                                                                                       val_c=c_value))
            for _ in range(cls.tasks):
                cls.put_key(ckey, True)

        # Every process waits here until go keys are available
        cls.get_key(ckey)

    @classmethod
    def get_key(cls, key):
        return cls.kvs_client.get(key)

    @classmethod
    def put_key(cls, key, value):
        return cls.kvs_client.put(key, value)


def ownCheck(kvs, rank, chunk=1, kvs_key='count'):
    """
    Generator
    :param kvs: KVSClient
        KVS object for server access
    :param chunk: int
        The size of the chunk given to each subprocess
    :param kvs_key: str
        The KVS key to increment (default is 'count')
    :yield: bool
        True if this process has dibs on whatever. False if some other process has claimed it first.
    """
    if rank == 0:
        kvs.put(kvs_key, 0)

    # Start at the baseline
    checks, lower, upper = 0, -1, -1

    while True:

        # Checks increments every loop
        # If it's greater than the upper bound, get a new lower bound from the KVS count
        # Set the new upper bound by adding chunk to lower
        # And then put the new upper bound back into KVS key

        if checks >= upper:
            lower = kvs.get(kvs_key)
            upper = lower + chunk
            kvs.put(kvs_key, upper)

        # Yield TRUE if this row belongs to this process and FALSE if it doesn't
        yield lower <= checks < upper
        checks += 1