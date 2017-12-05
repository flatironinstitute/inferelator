#!/bin/sh
# Script to run the inferelator if you don't have the slurm task manager

# Example: to run on b. subtilis, set the first arg to bsubtilis_bbsr_workflow_runner.py
INFERELATOR_RUNNER_SCRIPT=$1


REPOSRC=https://github.com/simonsfoundation/kvsstcp

LOCALREPO=$(pwd)/kvsstcp

# We do it this way so that we can abstract if from just git later on
LOCALREPO_VC_DIR=$LOCALREPO/.git

if [ ! -d $LOCALREPO_VC_DIR ]
then
    git clone $REPOSRC $LOCALREPO
else
    pushd $LOCALREPO
    git pull $REPOSRC
    popd
fi

export PYTHONPATH=$PYTHONPATH:$LOCALREPO

export SLURM_PROCID=0
export SLURM_NTASKS=1

python $LOCALREPO/kvsstcp.py --execcmd 'python $1'
