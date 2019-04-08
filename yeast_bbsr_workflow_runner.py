from inferelator import workflow
from inferelator.distributed.inferelator_mp import MPControl
from inferelator import utils

utils.Debug.set_verbose_level(1)

MPControl.set_multiprocess_engine("multiprocessing")
MPControl.client.processes = 3

wflow = workflow.inferelator_workflow(regression="bbsr", workflow="tfa")
# Common configuration parameters
wflow.input_dir = 'data/yeast'
wflow.priors_file = 'yeast-motif-prior.tsv'
wflow.multiprocessing_controller = "local"
wflow.num_bootstraps = 2
wflow.delTmax = 110
wflow.delTmin = 0
wflow.tau = 45

if __name__ == "__main__":
    wflow.run()

