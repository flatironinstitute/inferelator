from __future__ import print_function
import pandas as pd
import numpy as np
import os

# Get the following environment variables
# Workflow_variable_name, casting function, default (if the env isn't set or the casting fails for whatever reason)
SBATCH_VARS = {'RUNDIR': ('output_dir', str, None),
               'DATADIR': ('input_dir', str, None),
               'SLURM_PROCID': ('rank', int, 0),
               'SLURM_NTASKS_PER_NODE': ('cores', int, 1),
               'SLURM_NTASKS': ('tasks', int, 1),
               'SLURM_NODEID': ('node', int, 0),
               'SLURM_JOB_NUM_NODES': ('num_nodes', int, 1)
               }


def slurm_envs():
    envs = {}
    for os_var, (cv, mt, de) in SBATCH_VARS.items():
        try:
            val = mt(os.environ[os_var])
        except (KeyError, TypeError):
            val = de
        envs[cv] = val
    return envs


class Debug:
    verbose_level = 0
    default_level = 1

    silence_clients = True
    rank = slurm_envs()['rank']

    levels = dict(silent=-1,
                  normal=0,
                  verbose=1, v=1,
                  very_verbose=2, vv=2,
                  max_output=3, vvv=3)

    @classmethod
    def set_verbose_level(cls, lvl):
        if isinstance(lvl, (int, float)):
            cls.verbose_level = lvl

    @classmethod
    def vprint(cls, *args, **kwargs):
        if cls.silence_clients and cls.rank != 0:
            return
        cls.print_level(*args, **kwargs)

    @classmethod
    def warn(cls, *args, **kwargs):
        cls.vprint(*args, level=cls.levels["v"], **kwargs)

    @classmethod
    def notify(cls, *args, **kwargs):
        cls.vprint(*args, level=cls.levels["vv"], **kwargs)

    @classmethod
    def vprint_all(cls, *args, **kwargs):
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


def ownCheck(kvs, rank, chunk=1, kvs_key='count'):
    """
    Generator

    :param kvs: KVSClient
        KVS object for server access
    :param rank: int
        SLURM proc ID
    :param chunk: int
        The size of the chunk given to each subprocess
    :param kvs_key: str
        The KVS key to increment (default is 'count')

    :yield: bool
        True if this process has dibs on whatever. False if some other process has claimed it first.
    """

    # If we're the main process, set KVS key to 0
    if 0 == rank:
        kvs.put(kvs_key, 0)

    # Start at the baseline
    checks, lower, upper = 0, -1, -1

    while 1:

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


def kvsTearDown(kvs, rank, kvs_key='count'):
    # de-initialize the global counter.        
    if 0 == rank:
        # Do a hard reset if rank == 0                                                                                                       
        kvs.get(kvs_key)


def kvs_sync_processes(kvs, rank, pref=""):
    # Block all processes until they reach this point
    # Then release them
    # It may be wise to use unique prefixes if this is gonna get called rapidly so there's no collision
    # Or not. I'm a comment, not a cop.

    n = slurm_envs()['tasks']
    wkey = pref + '_wait'
    ckey = pref + '_continue'

    kvs.put(wkey, True)
    if rank == 0:
        for _ in range(n):
            kvs.get(wkey)
        for _ in range(n):
            kvs.put(ckey, True)
    kvs.get(ckey)


def kvs_sync_then_teardown(kvs, rank, kvs_key='count'):
    kvs_sync_processes(kvs, rank)
    kvsTearDown(kvs, rank, kvs_key=kvs_key)


class kvs_async:
    """
    Controller for process condition. Allows processes to synchronize at specific points or to execute workflow
    asynchronously.
    """

    master = False

    def __init__(self, kvs, chunk=1, pref='sync'):
        """
        Creates the kvs manager
        :param kvs: KVSClient
        :param chunk: int
            Number of processes to run at the same time
        :param pref: str
            Prefix for this kvs manager's KVS keys. Only needed if there's a couple of these running around the same
            time.
        """

        self.kvs = kvs
        self.chunk = chunk

        # Get rank & ntasks from environment variables
        envs = slurm_envs()
        self.n = envs['tasks']
        self.r = envs['rank']
        self.node = envs['node']
        self.num_nodes = envs['num_nodes']

        if self.r == 0:
            self.master = True

        # Create key names for KVS
        self.pref = pref
        self.nodekey = pref + "_node"
        self.startkey = pref + "_runner_"
        self.readykey = pref + "_ready"
        self.releasekey = pref + "_release"

    def execute_master_first(self, fun, *args, **kwargs):
        Debug.vprint("Running master setup")
        if self.master:
            fun(*args, **kwargs)
        Debug.vprint("Master setup complete")
        self.async_start()
        if not self.master:
            fun(*args, **kwargs)
        self.async_hold()

    def execute_async(self, fun, *args, **kwargs):
        """
        Execute function asynchronously and then block till everyone's finished. Pass all other arguments to the
        function

        :param fun: function
        """
        self.async_start()
        fun(*args, **kwargs)
        self.async_hold()

    def async_start(self):
        """
        Create the initial start keys if master. Otherwise sit here and wait.
        """
        self._put_node()
        if self.master:
            self._get_nodes()
            self._master_start()
        rkey = self.startkey + str(self.r)
        self.kvs.get(rkey)

    def async_hold(self):
        """
        Create a key to report that this process is complete. If master, create a new start key every time someone
        reports that they're complete. Once all the processes are complete, create release keys to allow everyone to
        move past this blocking point.
        """
        self.kvs.put(self.readykey, self.r)
        if self.master:
            self._master_control()
            self._master_release()
        self.kvs.get(self.releasekey)

    def sync_point(self):
        """
        Wrapper for kvs_sync_processes
        """
        kvs_sync_processes(self.kvs, self.r, pref=self.pref)

    def _master_start(self):
        """
        Start either chunk number of processes or ntasks number of processes (whichever is smaller)
        """
        for i in range(len(self.wait_nodes)):
            for _ in range(self.chunk):
                try:
                    # Make sure that the master process gets started
                    start = self.wait_nodes[i].pop(self.wait_nodes[i].index(0))
                except IndexError:
                    start = self.wait_nodes[i].pop()
                self._start_process(start)

    def _master_control(self):
        """
        Every time a process finishes, start another one until ntasks processes have been reached
        """
        for i in range(self.n):
            open_node = self.proc_map[self.kvs.get(self.readykey)]
            if len(self.wait_nodes[open_node]) > 0:
                self._start_process(self.wait_nodes[open_node].pop())

    def _master_release(self):
        """
        Put a release key onto KVS for every process
        """
        for i in range(self.n):
            self.kvs.put(self.releasekey, True)
        Debug.vprint("Asynchronous start complete. Releasing processes.")

    def _put_node(self):
        """
        Put a key onto KVS identifying which node this process is on
        """
        self.kvs.put(self.nodekey, (self.r, self.node))

    def _get_nodes(self):
        """
        Get the node keys off KVS to build a map of which processes are on which nodes
        :return:
        """
        self.proc_map = {}
        self.wait_nodes = [[]] * self.num_nodes
        for i in range(self.n):
            rank, node = self.kvs.get(self.nodekey)
            self.wait_nodes[node].append(rank)
            self.proc_map[rank] = node

    def _start_process(self, pid):
        key = self.startkey + str(pid)
        self.kvs.put(key, True)
        Debug.vprint("Starting process {rank}".format(rank=pid))



def df_from_tsv(file_like, has_index=True):
    "Read a tsv file or buffer with headers and row ids into a pandas dataframe."
    return pd.read_csv(file_like, sep="\t", header=0, index_col=0 if has_index else False)


def metadata_df(file_like):
    "Read a metadata file as a pandas data frame."
    return pd.read_csv(file_like, sep="\t", header=0, index_col="condName")


def read_tf_names(file_like):
    "Read transcription factor names from one-column tsv file.  Return list of names."
    exp = pd.read_csv(file_like, sep="\t", header=None)
    assert exp.shape[1] == 1, "transcription factor file should have one column "
    return list(exp[0])


def df_set_diag(df, val, copy=True):
    """
    Sets the diagonal of a dataframe to a value. Diagonal in this case is anything where row label == column label.

    :param df: pd.DataFrame
        DataFrame to modify
    :param val: numeric
        Value to insert into any cells where row label == column label
    :param copy: bool
        Force-copy the dataframe instead of modifying in place
    :return: pd.DataFrame / int
        Return either the modified dataframe (if copied) or the number of cells modified (if changed in-place)
    """

    # Find all the labels that are shared between rows and columns
    isect = df.index.intersection(df.columns)

    if copy:
        df = df.copy()

    # Set the value where row and column names are the same
    for i in range(len(isect)):
        df.loc[isect[i], isect[i]] = val

    if copy:
        return df
    else:
        return len(isect)


def bool_to_index(arr):
    """
    Returns an array that indexes all the True elements of a boolean array
    :param arr: np.ndarray
    :return: np.ndarray
    """
    return np.where(arr)[0]


def index_of_nonzeros(arr):
    """
    Returns an array that indexes all the non-zero elements of an array
    :param arr: np.ndarray
    :return: np.ndarray
    """
    return np.where(arr != 0)[0]


def make_array_2d(arr):
    """
    Changes array shape from 1d to 2d if needed (in-place)
    :param arr:  np.ndarray
    """
    if arr.ndim == 1:
        arr.shape = (arr.shape[0], 1)
