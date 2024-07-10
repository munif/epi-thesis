#!/bin/bash
#SBATCH --job-name=encoding_data
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=4
#SBATCH --partition=long
#SBATCH --ntasks=64
#SBATCH --time=2-00:00:00
#SBATCH --mem=500G

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/encoding_data.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"