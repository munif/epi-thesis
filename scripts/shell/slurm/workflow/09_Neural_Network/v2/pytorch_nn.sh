#!/bin/bash
#SBATCH --job-name=09-pytorch-nn
#SBATCH --output=/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/09/slurm-%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/09/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

module load Anaconda3/2024.06 cuda/12.4

# Record start time
start_time=$(date +%s.%N)

echo "Pytorch NN"

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/09_Neural_Network/pytorch_nn.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"