from inferelator.distributed.dask_local_controller import DaskController


class DaskK8Controller(DaskController):
    """
    The DaskK8Controller class launches a local dask cluster and connects as a client
    """

    _controller_name = "dask-k8"
