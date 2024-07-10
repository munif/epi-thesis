#!/bin/bash
#SBATCH --job-name=xgboost-cross-fold
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=4
#SBATCH --partition=work
#SBATCH --ntasks=64
#SBATCH --nodelist=n[010-013,023,029-031]
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/07_crossfold_xgboost.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"