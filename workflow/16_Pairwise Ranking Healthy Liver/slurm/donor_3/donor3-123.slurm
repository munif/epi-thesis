#!/bin/bash
#SBATCH --job-name=d3_ranking
#SBATCH --output="/group/pmc021/amunif/epi-thesis/workflow/16_Pairwise Ranking Healthy Liver/slurm/donor_3/output/%j-%x.out"
#SBATCH --error="/group/pmc021/amunif/epi-thesis/workflow/16_Pairwise Ranking Healthy Liver/slurm/donor_3/output/%j-%x.err"
#SBATCH --nodes=1
#SBATCH --partition=gpu
##SBATCH --partition=work
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB

# Load the Environment
module load Anaconda3/2024.06
module load gcc

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -p /group/pmc021/amunif/env/pytorch2 python "/group/pmc021/amunif/epi-thesis/workflow/16_Pairwise Ranking Healthy Liver/slurm/donor_3/donor3.py" 0 32 123

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"