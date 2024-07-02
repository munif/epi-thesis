#!/bin/bash
#SBATCH --job-name=09-pytorch-conv1d
#SBATCH --output=/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/09/slurm-%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/09/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Record start time
start_time=$(date +%s.%N)

echo "Pytorch Conv1D"

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/09_Neural_Network/pytorch_conv1d.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"