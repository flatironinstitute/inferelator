"""
LocalController just runs everything in a single process
"""

import itertools

from inferelator_ng.distributed import AbstractController, process_dask_graph, process_dask_function_args


class LocalController(AbstractController):
    client = None
    is_master = True

    @classmethod
    def connect(cls, *args, **kwargs):
        return True

    @classmethod
    def sync_processes(cls, *args, **kwargs):
        return True

    @classmethod
    def get(cls, dsk, result, *args, **kwargs):
        func, map_args, iter_args, iter_product = process_dask_graph(dsk, result)

        if map_args is None:
            return func()
        elif iter_args is None:
            return func(*map_args)

        # Just chew through the results all in order
        results = {pos: func(*process_dask_function_args(map_args, iter_args, iterated_args))
                   for pos, iterated_args in enumerate(itertools.product(*iter_product))}

        # Process results
        return cls.process_results(results)

    @classmethod
    def process_results(cls, results):
        # Put everything into a list based on the dict key
        pileup_list = [None] * len(results)
        for idx, val in results.items():
            pileup_list[idx] = val

        return pileup_list
