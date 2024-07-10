#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --output=/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/09/slurm-%x-%j.out
#SBATCH --error=/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/09/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1  # Request 1 GPU V100
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB

nvidia-smi