#!/bin/bash
#SBATCH --job-name=1000-xgboost
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/05_xgboost_with_train_test/slurm-%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/05_xgboost_with_train_test/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Record start time
start_time=$(date +%s.%N)

echo "XGBoost"

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/epi-thesis/workflow/05_xgboost_with_train_test/xgboost_1000.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"