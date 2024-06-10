#!/bin/bash
#SBATCH --job-name=hello-kaya
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --ntasks=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=10G

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -n epi-thesis python /group/pmc021/amunif/project/hello_world.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"
