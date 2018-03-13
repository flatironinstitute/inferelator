#!/bin/sh
# Script to run the inferelator if you don't have the slurm task manager
# Usage: ./inferelator_runner_slurmless.sh <specific_organism_runner>.py
# Example: to run on b. subtilis, set the first arg to bsubtilis_bbsr_workflow_runner.py

INFERELATOR_RUNNER_SCRIPT=$1

# Setting mock variables to replicate what Slurm Env variables
# would show when running 1 single process
export SLURM_PROCID=0
export SLURM_NTASKS=1

python -m kvsstcp.kvsstcp --execcmd "python $1"
