"""
KVSController is a wrapper for KVSClient that adds some useful functionality related to interprocess
communication.

It also keeps track of a bunch of SLURM related stuff that was previously workflow's problem.
"""

from kvsstcp import KVSClient

import os

# SLURM environment variables
SBATCH_VARS = dict(SLURM_PROCID=('rank', int, 0),
                   SLURM_NTASKS_PER_NODE=('cores', int, 1),
                   SLURM_NTASKS=('tasks', int, 1),
                   SLURM_NODEID=('node', int, 0),
                   SLURM_JOB_NUM_NODES=('num_nodes', int, 1))

NODE_MAP_KEY = "kvs_node_map"


class KVSController(KVSClient):
    # Set from SLURM environment variables
    rank = None  # int
    tasks = None  # int
    node = None  # int
    cores = None  # int
    num_nodes = None  # int
    is_master = False  # bool

    # Poll workers for location
    proc_nodes = list()  # list

    def __init__(self, slurm=True, **kwargs):

        # Get local environment variables
        self._get_env(allow_defaults=not slurm)

        # Connect to the host server by calling to KVSClient.__init__
        super(KVSController, self).__init__(**kwargs)

        # Map the process to nodes
        if slurm:
            self._create_node_map()

    """
    The following code polls environment variables and other nodes to get information needed to manage
    processes through KVS
    """

    def _get_env(self, allow_defaults=False):
        """

        :param allow_defaults: bool
            Use the default values for environment variables. Otherwise throw an error if not set
        :return:
        """
        for env_var, (class_var, func, default) in SBATCH_VARS.items():
            try:
                val = func(os.environ[env_var])
            except (KeyError, TypeError):
                val = default
                if not allow_defaults:
                    print("ENV {var} is not set. Using default {defa}".format(var=env_var, defa=default))
            setattr(self, class_var, val)
        if self.rank == 0:
            self.is_master = True
        else:
            self.is_master = False

    def _create_node_map(self):
        """
        Get the node keys off KVS to build a map of which processes are on which nodes
        """

        self.put(NODE_MAP_KEY, (self.rank, self.node))

        if self.is_master:
            self.proc_nodes = [[]] * self.num_nodes
            for i in range(self.tasks):
                r, n = self.get(NODE_MAP_KEY)
                self.proc_nodes[n].append(r)

    def own_check(self, chunk=1, kvs_key='count'):
        # Create the persistant key for ownCheck
        if self.is_master:
            self.put(kvs_key, 0)

        # Create an ownCheck generator and return it
        return ownCheck(self, chunk, kvs_key)

    def finish_own_check(self, kvs_key='count'):
        if self.is_master:
            self.get(kvs_key)

    def sync_processes(self, pref=""):
        # Block all processes until they reach this point
        # Then release them
        # It may be wise to use unique prefixes if this is gonna get called rapidly so there's no collision
        # Or not. I'm a comment, not a cop.

        wkey = pref + '_wait'
        ckey = pref + '_continue'

        # Every process puts a wait key up when it gets here
        self.put(wkey, True)

        # The master pulls down the wait keys until it has all of them
        # Then it puts up a go key for each process
        if self.is_master:
            for _ in range(self.tasks):
                self.get(wkey)
            for _ in range(self.tasks):
                self.put(ckey, True)

        # Every process waits here until go keys are available
        self.get(ckey)


def ownCheck(kvs, chunk=1, kvs_key='count'):
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
            print("Process {i} claiming {l}-{u} in {k}".format(i=kvs.rank, l=lower, u=upper, k=kvs_key))

        # Yield TRUE if this row belongs to this process and FALSE if it doesn't
        yield lower <= checks < upper
        checks += 1
