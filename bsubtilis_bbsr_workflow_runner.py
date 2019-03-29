from inferelator_ng import workflow

wflow = workflow.inferelator_workflow(regression="bbsr", workflow="tfa")
# Common configuration parameters
wflow.input_dir = 'data/bsubtilis'
wflow.multiprocessing_controller = "local"
wflow.num_bootstraps = 2
wflow.delTmax = 110
wflow.delTmin = 0
wflow.tau = 45

if __name__ == "__main__":
    wflow.run()
