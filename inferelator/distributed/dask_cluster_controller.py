from __future__ import absolute_import, division, print_function
import time
import os

from dask import distributed
from dask_jobqueue import SLURMCluster

from inferelator import utils
from inferelator.distributed import AbstractController

# Maintain python 2 compatibility
try:
    from itertools import izip as zip
except ImportError:
    pass

DEFAULT_CORES = 20
DEFAULT_MEM = '62GB'
DEFAULT_INTERFACE = 'ib0'
DEFAULT_WALLTIME = '1:00:00'

ENV_EXTRA = ['module purge',
             'export MKL_NUM_THREADS=1',
             'export OPENBLAS_NUM_THREADS=1',
             'export NUMEXPR_NUM_THREADS=1']

CONTROLLER_EXTRA = []

DEFAULT_MIN_CORES = 20
DEFAULT_MAX_CORES = 200
DEFAULT_ADAPT_INTERVAL = "1s"
DEFAULT_ADAPT_WAIT_COUNT = 5

DEFAULT_NYU_HEADER = ['#SBATCH --nodes=1', '#SBATCH --ntasks-per-node=1']

try:
    DEFAULT_LOCAL_DIR = os.environ['TMPDIR']
except KeyError:
    DEFAULT_LOCAL_DIR = 'dask-worker-space'


# This is the worst thing I've ever written
def memory_limit_0(command_template):
    cargs = command_template.split("--")
    newer_better_command_args = []
    for carg in cargs:
        if carg.startswith("memory-limit"):
            carg = "memory-limit 0 "
        newer_better_command_args.append(carg)
    return "--".join(newer_better_command_args)


# Or maybe this is?
def fix_header_for_nyu(header, new_lines):
    header_lines = header.split("\n")
    newer_better_header = []
    for head in header_lines:
        if head.startswith("#SBATCH --cpus-per-task"):
            newer_better_header.extend(new_lines)
        newer_better_header.append(head)
    return "\n".join(newer_better_header)


class DaskHPCClusterController(AbstractController):
    """
    The DaskHPCClusterController launches a HPC cluster and connects as a client. By default it uses the SLURM workload
    manager, but other workload managers may be used.
    #TODO: test drop-in replacement of other managers

    Many of the cluster-specific options are taken from class variables that can be set prior to calling .connect().
    #TODO: eventually figure out how to get rid of the ugly monkeypatching hacks for command headers

    The map functionality is deliberately not implemented; dask-specific multiprocessing functions are used instead
    """
    _controller_name = "dask-cluster"
    _controller_dask = True

    is_master = True
    client = None

    ## Dask controller variables ##

    # Cluster Controller
    cluster_controller_class = SLURMCluster
    hack_cluster_controller_for_NYU = True

    # The dask cluster object
    local_cluster = None

    # If 0, turn off the memory nanny
    worker_memory_limit = 0
    allowed_failures = 10

    # Controls for the adaptive parameter
    control_adaptive = False
    minimum_cores = DEFAULT_MIN_CORES
    maximum_cores = DEFAULT_MAX_CORES
    interval = DEFAULT_ADAPT_INTERVAL
    wait_count = DEFAULT_ADAPT_WAIT_COUNT

    # SLURM specific variables
    queue = None
    project = None
    walltime = DEFAULT_WALLTIME
    cores = DEFAULT_CORES
    processes = DEFAULT_CORES
    memory = DEFAULT_MEM
    job_cpu = DEFAULT_CORES
    job_mem = DEFAULT_MEM
    env_extra = ENV_EXTRA
    cluster_controller_options = CONTROLLER_EXTRA
    interface = DEFAULT_INTERFACE
    local_directory = DEFAULT_LOCAL_DIR

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup slurm cluster
        """

        # Create a slurm cluster with all the various class settings
        cls.local_cluster = cls.cluster_controller_class(queue=cls.queue, project=cls.project, walltime=cls.walltime,
                                                         job_cpu=cls.job_cpu, cores=cls.cores, processes=cls.processes,
                                                         job_mem=cls.job_mem, env_extra=cls.env_extra,
                                                         interface=cls.interface, local_directory=cls.local_directory,
                                                         memory=cls.memory, job_extra=cls.cluster_controller_options)

        # Deactivate the worker memory nanny
        if cls.worker_memory_limit == 0:
            cls.local_cluster._command_template = memory_limit_0(cls.local_cluster._command_template)

        # Rewrite the command headers so that the SLURM controller will work with the NYU prince cluster
        if cls.hack_cluster_controller_for_NYU:
            cls.local_cluster.job_header = fix_header_for_nyu(cls.local_cluster.job_header, DEFAULT_NYU_HEADER)

        if cls.control_adaptive:
            cls.local_cluster.adapt(minimum=cls.minimum_cores, maximum=cls.maximum_cores, interval=cls.interval,
                                    wait_count=cls.wait_count)
        else:
            cls.local_cluster.adapt(minimum=cls.maximum_cores, maximum=cls.maximum_cores, interval=cls.interval,
                                    wait_count=cls.wait_count)

        cls.local_cluster.scheduler.allowed_failures = cls.allowed_failures
        cls.client = distributed.Client(cls.local_cluster)

        return True

    @classmethod
    def shutdown(cls):
        cls.client.close()
        cls.local_cluster.close()

    @classmethod
    def map(cls, func, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def sync_processes(self, *args, **kwargs):
        """
        This is a thing for KVS. Just return True.
        """
        return True

    @classmethod
    def add_worker_env_line(cls, line):
        """
        Add a line to the worker environment declaration
        This gets put into the sbatch script for workers
        This can be used to load modules, activate conda, etc
        """

        cls.env_extra.append(line)

    @classmethod
    def is_dask(cls):
        """
        Block when something asks if this is a dask function until the workers are alive
        """

        if cls.local_cluster._count_active_workers() > 0:
            return True

        sleep_time = 0
        while cls.local_cluster._count_active_workers() == 0:
            time.sleep(1)
            if sleep_time % 60 == 0:
                utils.Debug.vprint("Awaiting workers ({sleep_time} seconds elapsed)".format(sleep_time=sleep_time),
                                   level=0)
            sleep_time += 1

        return True
