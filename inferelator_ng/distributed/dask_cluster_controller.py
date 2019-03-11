from __future__ import absolute_import, division, print_function

# Maintain python 2 compatibility
try:
    from itertools import izip as zip
except ImportError:
    pass

import logging

logger = logging.getLogger(__name__)

from inferelator_ng.distributed import AbstractController

import dask
from dask import distributed
from distributed.utils import format_bytes
from dask_jobqueue import SLURMCluster
from dask_jobqueue.slurm import slurm_format_bytes_ceil

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

        self.memory_limit = memory_limit

        super(SLURMCluster, self).__init__(config_name=config_name, **kwargs)

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

        logger.debug("Job script: \n %s" % self.job_script())

    @property
    def worker_process_memory(self):
        if self.memory_limit is None:
            memory_limit = self.worker_memory / self.worker_processes
        elif self.memory_limit == 0:
            return 0
        else:
            memory_limit = self.memory_limit
        mem = format_bytes(memory_limit)
        mem = mem.replace(' ', '')
        return mem


class DaskSLURMController(AbstractController):
    is_master = True

    client = None
    local_cluster = None

    # Dask controller variables

    worker_memory_limit = 0
    minimum_cores = 20
    maximum_cores = 200

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
    def name(cls):
        return "dask"

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Setup local cluster
        """

        # It is necessary to properly configure ~/.config/dask/jobqueue.yaml prior to running this
        cls.local_cluster = NYUSLURMCluster(queue=cls.queue, project=cls.project, walltime=cls.walltime,
                                            job_cpu=cls.job_cpu, cores=cls.cores, processes=cls.processes,
                                            job_mem=cls.job_mem, env_extra=cls.env_extra, interface=cls.interface,
                                            local_directory=cls.local_directory, memory=cls.memory,
                                            memory_limit=cls.worker_memory_limit)
        cls.local_cluster.adapt(minimum=cls.minimum_cores, maximum=cls.maximum_cores, interval='1s')
        cls.client = distributed.Client(cls.local_cluster)

        return True

    @classmethod
    def shutdown(cls):
        cls.client.close()
        cls.local_cluster.close()

    @classmethod
    def map(cls, func, *args, **kwargs):
        """
        Map a function across iterable(s) and return a list of results

        :param func: function
            Mappable function
        :param args: iterable
            Iterator(s)
        :param chunk: int
            The number of iterations to assign in blocks
        :return:
        """

        raise NotImplementedError

    @classmethod
    def sync_processes(self, *args, **kwargs):
        """
        This is a thing for KVS. Just return True.
        """
        return True
