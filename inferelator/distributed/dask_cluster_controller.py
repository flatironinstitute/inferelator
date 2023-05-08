import os
import math
import subprocess
import copy

from dask import distributed
from dask_jobqueue import SLURMCluster
from dask_jobqueue.slurm import SLURMJob

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.distributed.dask import DaskAbstract


_DEFAULT_NUM_JOBS = 1
_DEFAULT_THREADS_PER_WORKER = 1
_DEFAULT_WORKERS_PER_JOB = 20
_DEFAULT_MEM_PER_JOB = '62GB'
_DEFAULT_INTERFACE = 'ib0'
_DEFAULT_WALLTIME = '1:00:00'

_DEFAULT_ENV_EXTRA = [
    'module purge'
]

_THREAD_CONTROL_ENV = [
    'export MKL_NUM_THREADS={t}',
    'export OPENBLAS_NUM_THREADS={t}',
     'export NUMEXPR_NUM_THREADS={t}'
    ]

_DEFAULT_CONTROLLER_EXTRA = [
    '--nodes 1',
    '--ntasks-per-node 1'
]

_KNOWN_CONFIG = {

    "greene": {
        "_job_n_workers": 24,
        "_job_mem": "62GB",
        "_job_time": "48:00:00",
        "_interface": "ib0",
        "_job_extra_env_commands": copy.copy(_DEFAULT_ENV_EXTRA)
    },

    "rusty_ccb": {
        "_job_n_workers": 28,
        "_num_local_workers": 25,
        "_job_mem": "498GB",
        "_job_time": "48:00:00",
        "_interface": "ib0",
        "_queue": "ccb",
        "_job_extra_env_commands": copy.copy(_DEFAULT_ENV_EXTRA)
    },

    "rusty_preempt": {
        "_job_n_workers": 40,
        "_num_local_workers": 35,
        "_job_mem": "766GB",
        "_job_time": "48:00:00",
        "_interface": "ib0",
        "_queue": "preempt",
        "_job_extra_env_commands": copy.copy(_DEFAULT_ENV_EXTRA),
        "_job_slurm_commands": copy.copy(_DEFAULT_CONTROLLER_EXTRA) + [
            "--qos=preempt",
            "--constraint=info"
        ]
    },

    "rusty_rome": {
        "_job_n_workers": 128,
        "_num_local_workers": 112,
        "_job_mem": "990GB",
        "_job_time": "24:00:00",
        "_interface": "ib0",
        "_queue": "ccb",
        "_job_extra_env_commands": copy.copy(_DEFAULT_ENV_EXTRA),
        "_job_slurm_commands": copy.copy(_DEFAULT_CONTROLLER_EXTRA) + [
            "--constraint=rome"
        ]
    },
}

_DEFAULT_LOCAL_WORKER_CMD = "dask-worker"

try:
    _DEFAULT_LOCAL_DIR = os.environ['TMPDIR']
except KeyError:
    _DEFAULT_LOCAL_DIR = 'dask-worker-space'

try:
    _DEFAULT_SLURM_ID = os.environ['SLURM_JOB_ID']
except KeyError:
    _DEFAULT_SLURM_ID = "1"


# This is the worst thing I've ever written
def memory_limit_0(command_template):
    return "--".join(
        ["memory-limit 0 "
        if c.startswith("memory-limit")
        else c
        for c in command_template.split("--")]
    )


class SLURMJobNoMemLimit(SLURMJob):

    def __init__(self, *args, **kwargs):
        super(SLURMJobNoMemLimit, self).__init__(
            *args,
            **kwargs
        )

        # Remove memory limit in workers
        self._command_template = memory_limit_0(
            self._command_template
        )


class DaskHPCClusterController(DaskAbstract):
    """
    The DaskHPCClusterController launches a HPC cluster
    and connects as a client. By default it uses the SLURM workload
    manager, but other workload managers may be used.

    #TODO: test drop-in replacement of other managers

    Many of the cluster-specific options are taken from class variables
    that can be set prior to calling .connect().

    The map functionality is deliberately not implemented; dask-specific
    multiprocessing functions are used instead
    """

    _controller_name = "dask-cluster"
    _controller_dask = True
    _require_initialization = True

    client = None

    # Cluster Controller
    _cluster_controller_class = SLURMCluster

    # Scale & await parameters
    _await_all_workers = False
    _await_non_local = False
    _await_complete = False

    # The dask cluster object
    local_cluster = None
    _tracker = None

    # Should any local workers be started on this node
    _num_local_workers = 0
    _runaway_protection = 5
    _local_worker_command = _DEFAULT_LOCAL_WORKER_CMD

    # SLURM specific variables
    _queue = None
    _project = None
    _interface = _DEFAULT_INTERFACE
    _local_directory = _DEFAULT_LOCAL_DIR
    _log_directory = None

    # Job variables
    _job_n = _DEFAULT_NUM_JOBS
    _job_n_workers = _DEFAULT_WORKERS_PER_JOB
    _worker_n_threads = _DEFAULT_THREADS_PER_WORKER
    _job_mem = _DEFAULT_MEM_PER_JOB
    _job_time = _DEFAULT_WALLTIME
    _job_slurm_commands = copy.copy(_DEFAULT_CONTROLLER_EXTRA)
    _job_extra_env_commands = copy.copy(_DEFAULT_ENV_EXTRA)
    _job_threading_commands = copy.copy(_THREAD_CONTROL_ENV)

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup slurm cluster
        """

        if cls.client is None:

        # Create a slurm cluster with all the various class settings
            cls.local_cluster = cls._cluster_controller_class(
                queue=cls._queue,
                project=cls._project,
                interface=cls._interface,
                walltime=cls._job_time,
                job_cpu=cls._job_n_workers * cls._worker_n_threads,
                cores=cls._job_n_workers * cls._worker_n_threads,
                processes=cls._job_n_workers,
                job_mem=cls._job_mem,
                job_script_prologue=cls._config_env(),
                local_directory=cls._local_directory,
                memory=cls._job_mem,
                job_extra_directives=cls._job_slurm_commands,
                job_cls=SLURMJobNoMemLimit,
                **kwargs
            )

            cls.client = distributed.Client(
                cls.local_cluster,
                direct_to_workers=True
            )

            cls._add_local_node_workers(cls._num_local_workers)
            cls._tracker = WorkerTracker()

            utils.Debug.vprint(
                f"Dask dashboard active: {cls.client.dashboard_link}",
                level=0
            )

            cls._scale_jobs()

        return True

    @classmethod
    def shutdown(cls):
        cls.client.close()
        cls.local_cluster.close()

        cls.client = None
        cls.local_cluster = None

    @classmethod
    def use_default_configuration(
        cls,
        known_config,
        n_jobs=1
    ):
        """
        Load a known default cluster configuration

        :param known_config: A string with a valid cluster configuration.
            Currently implemented are "greene" (NYU) and
            "rusty_ccb" (Simons Foundation; Center for Computational Biology)
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
            raise ValueError(
                f"Configuration {known_config} is unknown"
            )

        utils.Debug.vprint(cls._config_str(), level=1)

    @classmethod
    def set_job_size_params(
        cls,
        n_jobs=None,
        n_cores_per_job=None,
        mem_per_job=None,
        walltime=None,
        n_workers_per_job=None,
        n_threads_per_worker=None
    ):
        """
        Set the job size parameters

        :param n_jobs: The number of jobs to start. Each job
            will have separate workers and memory allocated.
            This is in addition to the main (scheduler) job.
            For slurm, each job will be set with
            "#SBATCH --nodes=1"
        :type n_jobs: int
        :param n_cores_per_job: The number of workers to start
            per job.
        For SLURM, this is setting "#SBATCH --cpus-per-task"
        :type n_cores_per_job: int
        :param mem_per_job: The amount of memory to allocate
            per job. For SLURM, this is setting "#SBATCH --mem"
        :type mem_per_job: str, int
        :param walltime: The time limit per worker job.
        For SLURM, this is setting #SBATCH --time
        :type walltime: str
        :param n_workers_per_job: The number of worker jobs to start
            SLURM will allocate n_workers_per_job * n_threads_per_worker
            cores per job.
        :type n_workers_per_job: int
        :param n_threads_per_worker: The number of threads to give each
            worker job. SLURM will allocate n_workers_per_job *
            n_threads_per_worker cores per job.

        """

        check.argument_integer(n_jobs, allow_none=True)
        check.argument_integer(n_cores_per_job, allow_none=True)
        check.argument_integer(n_threads_per_worker, allow_none=True)

        cls.set_param("_job_n", n_jobs)
        cls.set_param("_job_n_workers", n_cores_per_job)
        cls.set_param("_job_mem", mem_per_job)
        cls.set_param("_job_time", walltime)
        cls.set_param("_worker_n_threads", n_threads_per_worker)
        cls.set_param("_job_n_workers", n_workers_per_job)

    @classmethod
    def set_cluster_params(
        cls,
        queue=None,
        project=None,
        interface=None,
        local_workers=None
    ):
        """
        Set parameters which are specific to the HPC environment

        :param queue: The name of the SLURM partition to use.
            If None, do not specify a partition. Defaults to None.
        :type queue: str
        :param project: The name of the project for each job.
            If None, do not specify a project. Defaults to None.
        :type project: str
        :param interface: A string that identifies the network interface to use.
            Possible options may include 'eth0' or 'ib0'.
        :type interface: str
        :param local_workers: The number of local workers to start
            on the same node as the main scheduler
        :type local_workers: int
        """

        cls.set_param("_queue", queue)
        cls.set_param("_project", project)
        cls.set_param("_interface", interface)
        cls.set_param("_num_local_workers", local_workers)

    @classmethod
    def set_processes(cls, process_count):
        """
        Set the number of dask workers to use
        :param process_count: int
        :return:
        """
        check.argument_integer(process_count, low=1)
        cls._job_n = math.ceil(process_count / cls._job_n_workers)

        utils.Debug.vprint(
            "Using `set_processes` is not advised for the "
            "DASK CLUSTER configuration, Using `set_job_size_params` "
            f"is highly preferred. Configured {cls._job_n} jobs with "
            f"{cls._job_n_workers} workers per job.",
            level=0
        )

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
        :param env: The environment to activate. This will be used with
            the $CONDA_PREFIX variables if no argument
        is provided for `cmd`.
        :type env: str
        """

        if cmd is None:
            if 'CONDA_PREFIX_1' in os.environ:
                conda = os.environ['CONDA_PREFIX_1']
            elif 'CONDA_PREFIX' in os.environ:
                conda = os.environ['CONDA_PREFIX']
            else:
                raise ValueError(
                    "Unable to set conda environment implicitly; "
                    "pass explicit `cmd=` argument instead"
                )

            cmd = os.path.join(conda, "bin", "activate")
            cmd = "source " + cmd + " " + (env if env is not None else "")

        elif env is not None:
            utils.Debug.vprint(
                "The `env` argument to `add_worker_conda` is ignored "
                "when `cmd` is passed",
                level=0
            )

        cls.add_worker_env_line(cmd)
        return True

    @classmethod
    def map(
        cls,
        *args,
        **kwargs
    ):
        cls.check_cluster_state()
        return super().map(*args, **kwargs)

    @classmethod
    def check_cluster_state(
        cls,
        skip_await=False
    ):
        """
        Check to make sure that workers have been provisioned.
        Sleep until workers become available.
        """

        cls._scale_jobs()

        _total_workers = cls._total_workers()

        # If there are no workers to be had, skip waiting
        if skip_await or _total_workers == 0:
            return

        # If we've already done this, just make sure
        # The cluster isn't dead
        elif cls._await_complete:
            _require_workers = 1

        # If we want all the workers up
        elif cls._await_all_workers:
            _require_workers = _total_workers

        # If we want at least one non-local worker up
        # Silently is ignored if there are no non-local workers
        elif cls._await_non_local:
            _require_workers = cls._num_local_workers

            if _require_workers is None:
                _require_workers = 0

            if _total_workers > _require_workers:
                _require_workers += 1

        # Otherwise just wait for at least one worker
        else:
            _require_workers = 1

        # Blocking call to distributed.client
        cls.client.wait_for_workers(
            n_workers=_require_workers
        )

        cls._await_complete = True


    @classmethod
    def _config_str(cls):

        return (
            f"Dask cluster: Allocated {cls._job_n} jobs ({cls._job_n_workers} "
            f"workers with {cls._job_mem} memory per job) "
            f"plus {cls._num_local_workers} local workers "
            f"[SLURM]: -p {cls._queue}, -A {cls._project}, "
            f"{', '.join(cls._job_slurm_commands)} "
            f"[ENV]: {', '.join(cls._job_extra_env_commands)}"
        )

    @classmethod
    def _config_env(cls):
        return [
            s.format(t=cls._worker_n_threads)
            for s in cls._job_threading_commands
        ] + cls._job_extra_env_commands

    @classmethod
    def _scale_jobs(cls):
        """
        Update the worker tracker. If an entire slurm job is dead,
        start a new one to replace it.
        """
        cls._tracker.update_lists(
            cls.local_cluster.observed,
            cls.local_cluster.worker_spec
        )

        new_jobs = cls._job_n + cls._tracker.num_dead
        max_jobs = cls._runaway_protection * cls._job_n

        if cls._runaway_protection is not None and new_jobs > max_jobs:
            raise RuntimeError(
                "Aborting excessive worker startups and "
                "protecting against runaway job queueing")
        elif new_jobs > len(cls.local_cluster.worker_spec):
            cls.local_cluster.scale(jobs=new_jobs)

    @classmethod
    def _add_local_node_workers(cls, num_workers):
        """
        Start workers on the local node with the scheduler & client

        :param num_workers: The number of workers to start on this node
        :type num_workers: int
        """
        check.argument_integer(
            num_workers,
            low=0,
            allow_none=True
        )

        if num_workers is not None and num_workers > 0:

            # Build a dask-worker command
            cmd = [
                cls._local_worker_command,
                str(cls.local_cluster.scheduler_address),
                "--nprocs", str(num_workers),
                "--nthreads", str(cls._worker_n_threads),
                "--memory-limit", "0",
                "--local-directory", str(cls._local_directory)
            ]

            # Execute it through the Popen ()
            if cls._log_directory is not None:
                out_path = cls._log_directory
            else:
                out_path = "."

            if not os.path.exists(out_path):
                os.makedirs(out_path, exist_ok=True)

            subprocess.Popen(
                cmd,
                stdout=open(
                    os.path.join(out_path, f"slurm-{_DEFAULT_SLURM_ID}.out"),
                    mode="w"
                ),
                stderr=open(
                    os.path.join(out_path, f"slurm-{_DEFAULT_SLURM_ID}.err"),
                    mode="w"
                )
            )

    @classmethod
    def _total_workers(
        cls,
        include_local = True
    ):
        """
        Get the total number of workers requested

        :param include_local: Include local workers in total,
            defaults to True
        :type include_local: bool, optional
        :return: Number of worker processes
        :rtype: int
        """

        _total = cls._job_n_workers if cls._job_n_workers is not None else 0
        _total *= cls._job_n if cls._job_n is not None else 0

        if include_local and cls._num_local_workers is not None:
            _total += cls._num_local_workers

        return _total


class WorkerTracker:
    """
    Keep track of which workers have been started but are now gone
    Workers with no live procs will be assumed dead
    """

    def __init__(self):
        self._live_workers = set()
        self._dead_workers = set()

        self._dead_cluster_job = set()

    def update_lists(self, current_alive, worker_spec):
        self._dead_workers.update(self._live_workers.difference(current_alive))
        self._live_workers.update(current_alive)

        for k, v in worker_spec.items():
            if k in self._dead_cluster_job:
                pass

            workers_in_spec = set(str(k) + g for g in v['group'])
            if workers_in_spec.issubset(self._dead_workers):
                self._dead_cluster_job.add(k)

    @property
    def num_dead(self):
        return len(self._dead_cluster_job)
