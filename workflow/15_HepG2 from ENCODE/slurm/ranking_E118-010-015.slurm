#!/bin/bash
#SBATCH --job-name=E118
#SBATCH --output="/group/pmc021/amunif/epi-thesis/workflow/15_HepG2 from ENCODE/slurm/output/%j-%x.out"
#SBATCH --error="/group/pmc021/amunif/epi-thesis/workflow/15_HepG2 from ENCODE/slurm/output/%j-%x.err"
#SBATCH --nodes=1
#SBATCH --partition=gpu
##SBATCH --partition=work
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=32GB

# Load the Environment
module load Anaconda3/2024.06
module load gcc

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -p /group/pmc021/amunif/env/pytorch2 python "/group/pmc021/amunif/epi-thesis/workflow/15_HepG2 from ENCODE/slurm/ranking_E118.py" 10 15

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"