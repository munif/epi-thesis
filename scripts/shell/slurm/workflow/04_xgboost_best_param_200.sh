#!/bin/bash
#SBATCH --job-name=200-xgboost-best-param
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=4
#SBATCH --partition=work
#SBATCH --ntasks=64
#SBATCH --nodelist=n[011-013,022-025,028-030,033]
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/04_xgboost_best_param_200.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"