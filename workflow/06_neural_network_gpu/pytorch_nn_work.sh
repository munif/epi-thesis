#!/bin/bash
#SBATCH --job-name=pytorch-nn
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/slurm-output/%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/slurm-output/%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Load the Environment
module load Anaconda3/2024.06

# Record start time
start_time=$(date +%s.%N)

echo "Pytorch Conv1D"

# Run Python script
conda run -p /group/pmc021/amunif/env/pytorch python /group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/pytorch_nn.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"