from __future__ import absolute_import, division, print_function
import time

import dask
from dask import distributed
from dask_jobqueue import SLURMCluster
from dask_jobqueue.slurm import slurm_format_bytes_ceil

from inferelator_ng.distributed import AbstractController

# Maintain python 2 compatibility
try:
    from itertools import izip as zip
except ImportError:
    pass

DEFAULT_CORES = 20
DEFAULT_MEM = '62GB'
DEFAULT_INTERFACE = 'ib0'
DEFAULT_LOCAL_DIR = '$TMPDIR'
DEFAULT_WALLTIME = '1:00:00'

ENV_EXTRA = ['module purge',
             'module load python/intel/2.7.13',
             'module load gcc/6.3.0',
             'export MKL_NUM_THREADS=1',
             'export OPENBLAS_NUM_THREADS=1',
             'export NUMEXPR_NUM_THREADS=1']

DEFAULT_MIN_CORES = 20
DEFAULT_MAX_CORES = 200
DEFAULT_ADAPT_INTERVAL = "1s"
DEFAULT_ADAPT_WAIT_COUNT = 5


# Overriding SLURMCluster to fix the hardcoded shit NYU hates
class NYUSLURMCluster(SLURMCluster):
    def __init__(self, queue=None, project=None, walltime=None, job_cpu=None, job_mem=None, job_extra=None,
                 config_name='slurm', memory_limit=None, **kwargs):
        if queue is None:
            queue = dask.config.get('jobqueue.%s.queue' % config_name)
        if project is None:
            project = dask.config.get('jobqueue.%s.project' % config_name)
        if walltime is None:
            walltime = dask.config.get('jobqueue.%s.walltime' % config_name)
        if job_cpu is None:
            job_cpu = dask.config.get('jobqueue.%s.job-cpu' % config_name)
        if job_mem is None:
            job_mem = dask.config.get('jobqueue.%s.job-mem' % config_name)
        if job_extra is None:
            job_extra = dask.config.get('jobqueue.%s.job-extra' % config_name)

        super(SLURMCluster, self).__init__(config_name=config_name, **kwargs)

        if memory_limit is not None and memory_limit == 0:
            self.memory_limit_0()

        # Always ask for only one task
        header_lines = []
        # SLURM header build
        if self.name is not None:
            header_lines.append('#SBATCH -J %s' % self.name)
        if self.log_directory is not None:
            header_lines.append('#SBATCH -e %s/%s-%%J.err' %
                                (self.log_directory, self.name or 'worker'))
            header_lines.append('#SBATCH -o %s/%s-%%J.out' %
                                (self.log_directory, self.name or 'worker'))
        if queue is not None:
            header_lines.append('#SBATCH -p %s' % queue)
        if project is not None:
            header_lines.append('#SBATCH -A %s' % project)

        # Init resources, always 1 task,
        # and then number of cpu is processes * threads if not set
        header_lines.append('#SBATCH --nodes=1')
        header_lines.append('#SBATCH --ntasks-per-node=1')
        header_lines.append('#SBATCH --cpus-per-task=%d' % (job_cpu or self.worker_cores))
        # Memory
        memory = job_mem
        if job_mem is None:
            memory = slurm_format_bytes_ceil(self.worker_memory)
        if memory is not None:
            header_lines.append('#SBATCH --mem=%s' % memory)

        if walltime is not None:
            header_lines.append('#SBATCH -t %s' % walltime)
        header_lines.extend(['#SBATCH %s' % arg for arg in job_extra])

        header_lines.append('JOB_ID=${SLURM_JOB_ID%;*}')

        # Declare class attribute that shall be overridden
        self.job_header = '\n'.join(header_lines)

    # This is the worst thing I've ever written
    def memory_limit_0(self):
        cargs = self._command_template.split("--")
        new_cargs = []
        for carg in cargs:
            if carg.startswith("memory-limit"):
                carg = "memory-limit 0 "
            new_cargs.append(carg)
        self._command_template = "--".join(new_cargs)


class DaskSLURMController(AbstractController):
    _controller_name = "dask"
    is_master = True
    client = None

    ## Dask controller variables ##

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
    interface = DEFAULT_INTERFACE
    local_directory = DEFAULT_LOCAL_DIR

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        cls.local_cluster = NYUSLURMCluster(queue=cls.queue, project=cls.project, walltime=cls.walltime,
                                            job_cpu=cls.job_cpu, cores=cls.cores, processes=cls.processes,
                                            job_mem=cls.job_mem, env_extra=cls.env_extra, interface=cls.interface,
                                            local_directory=cls.local_directory, memory=cls.memory,
                                            memory_limit=cls.worker_memory_limit)

        if cls.control_adaptive:
            cls.local_cluster.adapt(minimum=cls.minimum_cores, maximum=cls.maximum_cores, interval=cls.interval,
                                    wait_count=cls.wait_count)
        else:
            cls.local_cluster.adapt(minimum=cls.maximum_cores, maximum=cls.maximum_cores, interval=cls.interval,
                                    wait_count=cls.wait_count)

        sleep_time = 0
        while cls.local_cluster._count_active_workers() == 0:
            time.sleep(1)
            if sleep_time % 60 == 0:
                print("Awaiting workers ({sleep_time} seconds elapsed)".format(sleep_time=sleep_time))
            sleep_time += 1

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
