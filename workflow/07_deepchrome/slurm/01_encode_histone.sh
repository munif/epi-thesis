#!/bin/bash
#SBATCH --job-name=encode-histone
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/07_deepchrome/slurm/%j-%x.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/07_deepchrome/slurm/%j-%x.err
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Load the Environment
module load Anaconda3/2024.06

# Record start time
start_time=$(date +%s.%N)

echo "Encode histone"

# Run Python script
conda run -n epi-thesis python -u /group/pmc021/amunif/epi-thesis/workflow/07_deepchrome/slurm/01_encode_histone.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"