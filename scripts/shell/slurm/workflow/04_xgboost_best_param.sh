#!/bin/bash
#SBATCH --job-name=xgboost
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --ntasks=16
#SBATCH --nodelist=n[010-013]
#SBATCH --time=2-00:00:00
#SBATCH --mem=100GB

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/04_xgboost_best_param.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"