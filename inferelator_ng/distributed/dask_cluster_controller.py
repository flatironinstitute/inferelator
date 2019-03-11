import collections

# Maintain python 2 compatibility
try:
    from itertools import izip as zip
except ImportError:
    pass

from inferelator_ng.distributed import AbstractController
from inferelator_ng.utils import Validator as check

from dask import distributed
from dask_jobqueue import SLURMCluster

DEFAULT_CORES = 20
DEFAULT_MEM = '64GB'
DEFAULT_INTERFACE = 'ib0'
DEFAULT_LOCAL_DIR = '$TMPDIR'

ENV_EXTRA = ['module purge',
             'module load python/intel/2.7.13',
             'module load gcc/6.3.0',
             'export MKL_NUM_THREADS=1',
             'export OPENBLAS_NUM_THREADS=1',
             'export NUMEXPR_NUM_THREADS=1']


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
    walltime = None
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
        cls.local_cluster = SLURMCluster(queue=cls.queue, project=cls.project, walltime=cls.walltime,
                                         job_cpu=cls.job_cpu, cores=cls.cores, processes=cls.processes,
                                         job_mem=cls.job_mem, env_extra=cls.env_extra, interface=cls.interface,
                                         local_directory=cls.local_directory, memory= cls.memory,
                                         memory_limit = cls.worker_memory_limit)
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
