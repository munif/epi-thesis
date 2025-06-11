#!/bin/bash
#SBATCH --job-name=ranking_permutation
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/10_Permutation_Test_Ranking/slurm/output/%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/10_Permutation_Test_Ranking/slurm/output/%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
##SBATCH --partition=work
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

# Load the Environment
module load Anaconda3/2024.06

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -p /group/pmc021/amunif/env/pytorch python /group/pmc021/amunif/epi-thesis/workflow/10_Permutation_Test_Ranking/slurm/ranking_permutation.py 210 220

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"