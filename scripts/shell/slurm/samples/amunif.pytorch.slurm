#!/bin/bash
#SBATCH --job-name=pytorch-conv1d
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/slurm-output/%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/slurm-output/%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1  # Request 1 GPU V100
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Load the Environment
module load Anaconda3/2024.06
module load cuda/11.8

# activate the Python environment for the unit CITS5508
conda activate /group/pmc021/amunif/env/pytorch

conda list 
module list
# list the environment loaded by the modules.
# Can remove the two lines below if you want.
# module list
# conda list

# Note: SLURM_JOBID is a unique number for every job.
# These are generic variables.

# Below is the Python file that would be run.
SCRIPT=pytorch_conv1d_v1.py
PROJECT_DIR=${MYGROUP}/epi-thesis/workflow/06_neural_network_gpu
SCRATCH=${MYSCRATCH}/run_pytorch/${SLURM_JOBID}
RESULTS=${MYGROUP}/pytorch_results

##############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then 
    mkdir -p $SCRATCH 
fi 
echo Working SCRATCH directory is $SCRATCH

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then 
     mkdir -p $RESULTS
fi 
echo Results will be stored in $RESULTS/${SLURM_JOBID}

###############################################
# COPY the SCRIPT and any data to ${SCRATCH}
#  ALL JOBS MUST BE RUN IN $SCRATCH 
#

cp ${PROJECT_DIR}/${SCRIPT} ${SCRATCH}/.

# cp data ${SCRATCH}/.

cd ${SCRATCH}
# Record start time
start_time=$(date +%s.%N)

echo "Pytorch Conv1D"

# Run Python script

python ./${SCRIPT}

# Record end time
end_time=$(date +%s.%N)

#############################################
# Now move the output produced by your Python script from
# the /scratch... directory to your /group directory.
cd $HOME
mv ${SCRATCH} ${RESULTS}

echo "mv ${SCRATCH} ${RESULTS}"
echo "Please see the ${RESULTS}/${SLURM_JOBID} directory for any output"

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"