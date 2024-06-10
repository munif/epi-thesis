#!/bin/bash
#SBATCH --job-name=normvsauto
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --ntasks=35
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -n bdao python /group/sbs007/bdao/project/scripts/cpu/normvsauto.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"