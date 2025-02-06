#!/bin/bash
#SBATCH --job-name=does
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/09_Learning_to_Rank/slurm/all_features/output/%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/09_Learning_to_Rank/slurm/all_features/output/%x-%j.err
#SBATCH --nodes=1

#SBATCH --partition=gpu
##SBATCH --partition=work

##SBATCH --gres=gpu:p100:1  # Request 1 GPU V100 or P100
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=200GB
#SBATCH --signal=B:SIGTERM@60  # Send SIGTERM 60 seconds before timeout

# Signal handler to resubmit the job
resubmit_handler() {
    echo "Caught SIGTERM signal. Resubmitting job..."
    sbatch "$0"  # Resubmit the current script
    exit 0       # Exit gracefully
}

trap 'resubmit_handler' SIGTERM  # Trap SIGTERM to call handler

# Load the Environment
module load Anaconda3/2024.06

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -p /group/pmc021/amunif/env/pytorch python /group/pmc021/amunif/epi-thesis/workflow/09_Learning_to_Rank/slurm/all_features/do-earlystop.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"