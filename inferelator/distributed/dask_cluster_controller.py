from __future__ import absolute_import, division, print_function
import time
import os
import math

from dask import distributed
from dask_jobqueue import SLURMCluster
from dask_jobqueue.slurm import SLURMJob

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.distributed import AbstractController

_DEFAULT_CONDA_ACTIVATE = "source ~/.local/anaconda3/bin/activate"
_DEFAULT_NUM_JOBS = 1
_DEFAULT_WORKERS_PER_JOB = 20
_DEFAULT_MEM_PER_JOB = '62GB'
_DEFAULT_INTERFACE = 'ib0'
_DEFAULT_WALLTIME = '1:00:00'

_DEFAULT_ENV_EXTRA = ['module purge',
                      'export MKL_NUM_THREADS=1',
                      'export OPENBLAS_NUM_THREADS=1',
                      'export NUMEXPR_NUM_THREADS=1']

_DEFAULT_CONTROLLER_EXTRA = ['--nodes 1', '--ntasks-per-node 1']

_DEFAULT_ADAPTIVE_INTERVAL = "1s"

_KNOWN_CONFIG = {"prince": {"_job_n_workers": 20,
                            "_job_mem": "62GB",
                            "_job_time": "48:00:00",
                            "_interface": "ib0",
                            "_job_extra_env_commands": _DEFAULT_ENV_EXTRA
                            },
                 "rusty_ccb": {"_job_n_workers": 28,
                               "_job_mem": "498GB",
                               "_job_time": "48:00:00",
                               "_interface": "ib0",
                               "_queue": "ccb",
                               "_job_extra_env_commands": _DEFAULT_ENV_EXTRA}
                 }

try:
    _DEFAULT_LOCAL_DIR = os.environ['TMPDIR']
except KeyError:
    _DEFAULT_LOCAL_DIR = 'dask-worker-space'


# This is the worst thing I've ever written
def memory_limit_0(command_template):
    return "--".join(["memory-limit 0 " if c.startswith("memory-limit") else c for c in command_template.split("--")])


class SLURMJobNoMemLimit(SLURMJob):
    def __init__(self, *args, **kwargs):
        super(SLURMJobNoMemLimit, self).__init__(*args, **kwargs)
        self._command_template = memory_limit_0(self._command_template)


class DaskHPCClusterController(AbstractController):
    """
    The DaskHPCClusterController launches a HPC cluster and connects as a client. By default it uses the SLURM workload
    manager, but other workload managers may be used.
    #TODO: test drop-in replacement of other managers

    Many of the cluster-specific options are taken from class variables that can be set prior to calling .connect().

    The map functionality is deliberately not implemented; dask-specific multiprocessing functions are used instead
    """
    _controller_name = "dask-cluster"
    _controller_dask = True

    is_master = True
    client = None

    # Cluster Controller
    _cluster_controller_class = SLURMCluster

    # The dask cluster object
    _local_cluster = None

    # Controls for the adaptive parameter
    _adapt_interval = _DEFAULT_ADAPTIVE_INTERVAL

    # SLURM specific variables
    _queue = None
    _project = None
    _interface = _DEFAULT_INTERFACE
    _local_directory = _DEFAULT_LOCAL_DIR

    # Job variables
    _job_n = _DEFAULT_NUM_JOBS
    _job_n_workers = _DEFAULT_WORKERS_PER_JOB
    _job_mem = _DEFAULT_MEM_PER_JOB
    _job_time = _DEFAULT_WALLTIME
    _job_slurm_commands = _DEFAULT_CONTROLLER_EXTRA
    _job_extra_env_commands = _DEFAULT_ENV_EXTRA


    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup slurm cluster
        """

        # Create a slurm cluster with all the various class settings
        cls._local_cluster = cls._cluster_controller_class(queue=cls._queue,
                                                           project=cls._project,
                                                           interface=cls._interface,
                                                           walltime=cls._job_time,
                                                           job_cpu=cls._job_n_workers,
                                                           cores=cls._job_n_workers,
                                                           processes=cls._job_n_workers,
                                                           job_mem=cls._job_mem,
                                                           env_extra=cls._job_extra_env_commands,
                                                           local_directory=cls._local_directory,
                                                           memory=cls._job_mem,
                                                           job_extra=cls._job_slurm_commands,
                                                           job_cls=SLURMJobNoMemLimit)

        cls.client = distributed.Client(cls._local_cluster, direct_to_workers=True)
        return True

    @classmethod
    def shutdown(cls):
        cls.client.close()
        cls._local_cluster.close()

    @classmethod
    def map(cls, func, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def use_default_configuration(cls, known_config, n_jobs=1):
        """
        Load a known default cluster configuration

        :param known_config: A string with a valid known cluster configuration.
        Currently implemented are "prince" (NYU) and "rusty_ccb" (Simons Foundation; Center for Computational Biology)
        :type known_config: str
        :param n_jobs: Number of jobs to start with this configuration
        :type n_jobs: int
        """

        known_config = known_config.lower()
        if known_config in _KNOWN_CONFIG:
            for k, v in _KNOWN_CONFIG[known_config].items():
                setattr(cls, k, v)
            cls._job_n = n_jobs
        else:
            msg = "Configuration {k} is unknown".format(k=known_config)
            raise ValueError(msg)

        utils.Debug.vprint(cls._config_str(), level=1)

    @classmethod
    def set_job_size_params(cls, n_jobs=None, n_cores_per_job=None, mem_per_job=None, walltime=None):
        """
        Set the job size parameters

        :param n_jobs: The number of jobs to start. Each job will have separate workers and memory allocated.
        This is in addition to the main (scheduler) job.
        For slurm, each job will be set with "#SBATCH --nodes=1"
        :type n_jobs: int
        :param n_cores_per_job: The number of workers to start per job.
        For SLURM, this is setting "#SBATCH --cpus-per-task"
        :type n_cores_per_job: int
        :param mem_per_job: The amount of memory to allocate per job.
        For SLURM, this is setting "#SBATCH --mem"
        :type mem_per_job: str, int
        :param walltime: The time limit per worker job.
        For SLURM, this is setting #SBATCH --time
        :type walltime: str
        """

        check.argument_integer(n_jobs, allow_none=True)
        check.argument_integer(n_cores_per_job, allow_none=True)

        cls._job_n = n_jobs if n_jobs is not None else cls._job_n
        cls._job_n_workers = n_cores_per_job if n_cores_per_job is not None else cls._job_n_workers
        cls._job_mem = mem_per_job if mem_per_job is not None else cls._job_mem
        cls._job_time = walltime if walltime is not None else cls._job_time

    @classmethod
    def set_cluster_params(cls, queue=None, project=None, interface=None):
        """
        Set parameters which are specific to the HPC environment

        :param queue: The name of the SLURM partition to use. If None, do not specify a partition. Defaults to None.
        :type queue: str
        :param project: The name of the project for each job. If None, do not specify a project. Defaults to None.
        :type project: str
        :param interface: A string that identifies the network interface to use.
        Possible options may include 'eth0' or 'ib0'.
        :type interface: str
        """

        cls._queue = queue if queue is not None else cls._queue
        cls._project = project if project is not None else cls._project
        cls._interface = interface if interface is not None else cls._interface

    @classmethod
    def set_processes(cls, process_count):
        """
        Set the number of dask workers to use
        :param process_count: int
        :return:
        """
        check.argument_integer(process_count, low=1)
        cls._job_n = math.ceil(process_count / cls._job_n_workers)

        utils.Debug.vprint("Using `set_processes` is not advised for the DASK CLUSTER configuration", level=0)
        utils.Debug.vprint("Using `set_job_size_params` is highly preferred", level=0)
        utils.Debug.vprint("Configured {n} jobs with {w} workers per job".format(n=cls._job_n, w=cls._job_n_workers),
                           level=0)

    @classmethod
    def sync_processes(cls, *args, **kwargs):
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

        cls._job_extra_env_commands.append(line)

    @classmethod
    def add_slurm_command_line(cls, line):
        """
        Add a line to the command block for SLURM.
        This will be prepended with the "#SBATCH " command
        """

        cls._job_slurm_commands.append(line)

    @classmethod
    def add_worker_conda(cls, cmd=None, env=None):
        """
        Add a line to activate a conda environment for workers

        :param cmd: A shell command which will activate a conda environment
        Defaults to "source ~/.local/anaconda3/bin/activate base"
        :type cmd: str
        :param env: The environment to activate. This will be used with the $CONDA_PREFIX variables if no argument
        is provided for `cmd`.
        :type env: str
        """

        if cmd is None:
            try:
                conda = os.environ['CONDA_PREFIX_1'] if 'CONDA_PREFIX_1' in os.environ else os.environ['CONDA_PREFIX']
                cmd = os.path.join(conda, "bin", "activate")
                cmd = "source " + cmd + " " + (env if env is not None else "")
                cls.add_worker_env_line(cmd)
                return True
            except KeyError:
                raise ValueError("Unable to set conda environment implicitly; pass explicit `cmd=` argument instead")
        elif env is not None:
            utils.Debug.vprint("The `env` argument to `add_worker_conda` is ignored when `cmd` is passed", level=0)

        cls.add_worker_env_line(cmd)
        return True

    @classmethod
    def is_dask(cls):
        """
        Block when something asks if this is a dask function until the workers are alive
        """

        cls.check_cluster_state()
        return True

    @classmethod
    def check_cluster_state(cls):

        cls._scale_jobs()

        sleep_time = 0
        while len(cls._local_cluster.scheduler.identity()['workers']) == 0:
            time.sleep(1)
            if sleep_time % 60 == 0:
                utils.Debug.vprint("Awaiting workers ({sleep_time} seconds elapsed)".format(sleep_time=sleep_time),
                                   level=0)
            sleep_time += 1

    @classmethod
    def _config_str(cls):
        status = "\n".join(["Dask cluster: Allocated {n} jobs ({w} workers with {m} memory per job)",
                            "SLURM: -p {q}, -A {p}, " + ", ".join(cls._job_slurm_commands),
                            "ENV: " + "\n\t".join(cls._job_extra_env_commands)]) + "\n"
        return status.format(n=cls._job_n, w=cls._job_n_workers, m=cls._job_mem, q=cls._queue, p=cls._project)

    @classmethod
    def _scale_jobs(cls):
        expected_workers = cls._job_n * cls._job_n_workers
        if len(cls._local_cluster.scheduler.identity()['workers']) < expected_workers:
            cls._local_cluster.scale(jobs=cls._job_n)
