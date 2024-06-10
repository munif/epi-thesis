module avail
module load Anaconda3/2023.09
module list

# conda create --prefix /group/pmc021/amunif/conda_environments/epi-thesis
conda create -n munif python=3.11.8

conda init

# conda activate /group/pmc021/amunif/conda_environments/epi-thesis
conda activate epi-thesis

######################
srun --time=00:30:00 --account=amunif --partition=gpu --gres=gpu:p100:1 --pty /bin/bash -l


### Inside the GPU worker
module load Anaconda3/2023.09
conda create -n gpu_test python=3.11.5
conda activate gpu_test
pip install tensorflow[and-cuda]
