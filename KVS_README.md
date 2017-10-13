
#to run bbsr on the cluster, we use the kvs python library created by Dylan and Nick Carriero
#this new pipeline should also be able to run locally or on a non Slurm cluster 

################
#on slurm cluster

#First we allocate the number of nodes we want for the job, in this case 2
`srun -N 2 --exclusive --pty bash -i`

#Then we need the relevant python and R versions with the required libraries to run BBSR (you may already have the correct versions in your home directory)
```
export PATH=/mnt/xfs1/bioinfoCentos7/software/installs/python/anaconda/bin:$PATH
export PYTHONPATH=~dylan/scc/kvsstcp:/mnt/xfs1/bioinfoCentos7/software/installs/python/anaconda/bin:$PYTHONPATH
export R_LIBS="/mnt/xfs1/home/ndeveaux/R/x86_64-redhat-linux-gnu-library/3.3" 
```

#This is how we run bbsr on the cluster, this will also print the time it takes to run. Note that you can change the number of processes (as described by -n), 
#This is the parallelization parameter. For a bigger dataset like a yeast or human network, might set n to 56 for the srun of 2 nodes (28 cores X 2 nodes)
`time python ~dylan/scc/kvsstcp/kvsstcp.py --execcmd 'srun -n 8 python bsubtilis_bbsr_workflow_runner.py'`


################
#run kvs locally or on non Slurm Cluster

#We still need the same version of python and R but will also need to have KVS installed
#Locally, we will use the fauxSrun file rather than srun
#Below, we specify that we want to run 8 parallel processes at once (the argument after fauxSrun)
`time python <path to kvsstcp>/kvsstcp/kvsstcp.py --execcmd 'inferelator_ng/fauxSrun 8 python bsubtilis_bbsr_workflow_runner.py'`
