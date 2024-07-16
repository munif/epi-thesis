#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/slurm-output/slurm-%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1  # Request 1 GPU
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Load the Environment
module load Anaconda3/2024.06
module load cuda/11.8

# Record start time
start_time=$(date +%s.%N)

nvidia-smi

# Run Python script
conda run -p /group/pmc021/amunif/env/pytorch python /group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/test.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"