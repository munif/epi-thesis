#!/bin/bash
#SBATCH --job-name=r_d4
#SBATCH --output="/group/pmc021/amunif/epi-thesis/workflow/13_Healthy liver with manual preprocessing/slurm/donor_4/output/%j-%x.out"
#SBATCH --error="/group/pmc021/amunif/epi-thesis/workflow/13_Healthy liver with manual preprocessing/slurm/donor_4/output/%j-%x.err"
#SBATCH --nodes=1
#SBATCH --partition=gpu
##SBATCH --partition=work
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB

# Load the Environment
module load Anaconda3/2024.06
module load gcc

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -p /group/pmc021/amunif/env/pytorch2 python "/group/pmc021/amunif/epi-thesis/workflow/13_Healthy liver with manual preprocessing/slurm/donor_4/ranking_donor4.py" 15 20

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"