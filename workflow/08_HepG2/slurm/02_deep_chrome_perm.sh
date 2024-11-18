#!/bin/bash
#SBATCH --job-name=deepchrome
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/08_HepG2/slurm/output/%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/08_HepG2/slurm/output/%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
##SBATCH --partition=work
#SBATCH --gres=gpu:p100:1  # Request 1 GPU V100 or P100

#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Load the Environment
module load Anaconda3/2024.06

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -p /group/pmc021/amunif/env/pytorch python /group/pmc021/amunif/epi-thesis/workflow/08_HepG2/slurm/02_deep_chrome_perm.py 300 350

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"