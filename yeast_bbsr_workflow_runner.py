from inferelator_ng import workflow
from inferelator_ng.distributed.inferelator_mp import MPControl
from inferelator_ng import utils

utils.Debug.set_verbose_level(1)
MPControl.set_multiprocess_engine("dask-local")
MPControl.client.processes = 1
MPControl.client.local_dir = '/tmp'
MPControl.connect()

wflow = workflow.inferelator_workflow(regression="bbsr", workflow="tfa")
# Common configuration parameters
wflow.input_dir = '~/PycharmProjects/inferelator_sc/data'
wflow.append_to_path('input_dir', 'bsubtilis')
wflow.num_bootstraps = 2
wflow.delTmax = 110
wflow.delTmin = 0
wflow.tau = 45
wflow.gold_standard_filter_method = 'keep_all_gold_standard'

if __name__ == "__main__":
    wflow.run()
