#!/bin/bash
#
#SBATCH	--verbose
#SBATCH	--job-name=inferelator
#SBATCH	--output=inferelator-%j.out
#SBATCH	--error=inferelator-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32000
#SBATCH --time 48:00:00

# Clear environment modules
module purge

# Activate conda environment
source ~/anaconda3/bin/activate

# Activate module environment
# module load python

# Set environment variables to point to the data files and an output path
export RUNDIR=${SCRATCH}/inferelator/run-${SLURM_JOB_ID}/
export DATADIR=${SCRATCH}/inferelator/data

# Turn off buffering for stdout
export PYTHONUNBUFFERED=TRUE

# Control multithreading through environment flags
# numpy can be compiled with MKL
export MKL_NUM_THREADS=1
# numpy can also be compiled with OPENBLAS
export OPENBLAS_NUM_THREADS=1
# Google says this might also be a flag
export NUMEXPR_NUM_THREADS=1

# Print the principal slurm variables, the version of python and the major packages
echo "SLURM Environment: ${SLURM_JOB_NUM_NODES} Nodes ${SLURM_NTASKS} Tasks ${SLURM_MEM_PER_NODE} Memory/Node"
python -V
python -c "import numpy; print('NUMPY: ' + numpy.__version__)"
python -c "import pandas; print('PANDAS: ' + pandas.__version__)"

# Create the output path and copy the run script to it
echo "Creating run directory ${RUNDIR}"
mkdir -p ${RUNDIR}
cp ${1} ${RUNDIR}

# Execute the run script
time python ${1}
