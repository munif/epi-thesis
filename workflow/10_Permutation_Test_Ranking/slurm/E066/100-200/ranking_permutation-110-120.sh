#!/bin/bash
#SBATCH --job-name=RP_E066
#SBATCH --output=/group/pmc021/amunif/epi-thesis/workflow/10_Permutation_Test_Ranking/slurm/E066/output/%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/workflow/10_Permutation_Test_Ranking/slurm/E066/output/%x-%j.err
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
conda run -p /group/pmc021/amunif/env/pytorch python /group/pmc021/amunif/epi-thesis/workflow/10_Permutation_Test_Ranking/slurm/E066/ranking_permutation_E066.py 110 120

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"