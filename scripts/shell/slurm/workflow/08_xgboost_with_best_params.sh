#!/bin/bash
#SBATCH --job-name=08-xgboost
#SBATCH --output=/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/08/slurm-%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/08/slurm-%x-%j.err
#SBATCH --nodes=4
#SBATCH --partition=work
#SBATCH --ntasks=64
#SBATCH --nodelist=n[011-013,022-023,029-031,033]
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Record start time
start_time=$(date +%s.%N)

echo "XGBoost with n_estimators = 10000"

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/08_xgboost_with_best_params.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"