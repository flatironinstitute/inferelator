"""
KVSController is a wrapper for KVSClient that adds some useful functionality related to interprocess
communication.

It also keeps track of a bunch of SLURM related stuff that was previously workflow's problem.
"""


from kvsstcp import KVSClient

import os
import copy

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

    # Which keys are on the server that need to be removed later
    persist_keys = list() # list

    # Async settings
    pref = "async"
    chunk = 1
    master_first = False

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

        :param use_defaults:
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
            self.put_persistant_key(kvs_key, 0)

        # Create an ownCheck generator and return it
        return ownCheck(self, chunk, kvs_key)

    def finish_own_check(self, kvs_key='count'):
        if self.is_master:
            self.get_persistant_key(kvs_key)

    """
    Persistant keys are keys that are created by one process with no explicit kvs.get to remove them
    in the code (usually one process putting up data the others will view).
    
    This is management for these (so that they're easy to remove later).
    
    TODO: Implement a non-blocking delete that won't require any state knowledge
    """


    def put_persistant_key(self, key, value, encoding=True):
        if key in self.persist_keys:
            raise ValueError("Persistant key {k} in use".format(k=key))
        self.put(key, value, encoding)
        self.persist_keys.append(key)

    def get_persistant_key(self, *args):
        if len(self.persist_keys) == 0:
            return None

        data = list()
        for k in args:
            if k in self.persist_keys:
                data.append(self.get(k))
                self.persist_keys.remove(k)
        return data

    def clear_persistant_keys(self):
        if len(self.persist_keys) == 0:
            return None
        return [self.get(self.persist_keys.pop()) for _ in range(len(self.persist_keys))]

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

    """
    Everything under this is code to manage asynchronous process starts.
    This can be used to mitigate memory spiking
    """

    def async_settings(self, **kwargs):
        """
        :param master_first: bool
            Run everything on the master process and then do an async start
        :param pref: str
            String prefix for this async run
        :param chunk: int
            Number of processes per node to start
        """

        _async_settings = ["pref", "chunk", "master_first"]
        for k, v in kwargs.items():
            if k in _async_settings:
                setattr(self, k, v)
            else:
                raise ValueError("Not a valid argument to async_settings")
        return self

    def execute_async(self, fun, *args, **kwargs):
        """
        Execute function asynchronously and then block till everyone's finished.

        Pass all other arguments to the function

        :param fun: function
            Function to execute
        """

        if self.master_first and self.is_master:
            fun(*args, **kwargs)
            self._async_start()
        else:
            self._async_start()
            fun(*args, **kwargs)

        self._async_hold()

    def _async_start(self):
        """
        Sit here and wait.
        """
        if self.is_master:
            self._master_start()
        self.get(self._async_start_key())

    def _async_hold(self):
        """
        Create a key to report that this process is complete. If master, create a new start key every time someone
        reports that they're complete. Once all the processes are complete, create release keys to allow everyone to
        move past this blocking point.
        """
        self.put(self._async_ready_key(), self.rank)
        if self.is_master:
            self._master_controller()
        self.get(self._async_release_key())

    def _master_start(self):
        """
        Start either chunk number of processes on each node (whichever is smaller)
        """
        self.wait_nodes = copy.copy(self.proc_nodes)
        for i in range(self.num_nodes):
            for _ in range(min(self.chunk, len(self.proc_nodes[i]))):
                try:
                    # Make sure that the master process gets started
                    start = self.wait_nodes[i].pop(self.wait_nodes[i].index(0))
                except IndexError:
                    start = self.wait_nodes[i].pop()
                self._async_start_process(start)

    def _master_controller(self):
        """
        Every time a process finishes, start another one until ntasks processes have been reached
        """
        for i in range(self.tasks):
            open_node = self._which_node(self.get(self._async_ready_key()))
            if len(self.wait_nodes[open_node]) > 0:
                self._async_start_process(self.wait_nodes[open_node].pop())
        for i in range(self.tasks):
            self.put(self._async_release_key(), True)

    def _async_start_process(self, pid):
        """
        Start a process with a specific ID
        :param pid: int
        """
        self.put(self._async_start_key(pid), True)

    def _async_start_key(self, rank=None):
        if rank is None:
            return self.pref + "_start_" + str(self.rank)
        else:
            return self.pref + "_start_" + str(rank)

    def _async_ready_key(self):
        return self.pref + "_ready"

    def _async_release_key(self):
        return self.pref + "_release"

    def _which_node(self, procid):
        for i in range(len(self.proc_nodes)):
            if procid in self.proc_nodes[i]:
                return i
        raise ValueError("Process ID Unknown")


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

        # Yield TRUE if this row belongs to this process and FALSE if it doesn't
        yield lower <= checks < upper
        checks += 1
