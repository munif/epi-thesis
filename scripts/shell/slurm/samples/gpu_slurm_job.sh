#!/bin/bash
#SBATCH --job-name=gpu_normvsauto
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --gres=gpu:1  # Request 1 GPU V100
#SBATCH --partition=gpu  
#SBATCH --time=20:00:00
#SBATCH --mem=200G

# Load necessary modules
module load cuda  # Adjust according to the CUDA version installed

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -n rapids_env python /group/sbs007/bdao/project/scripts/gpu/gpu_normvsauto.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"