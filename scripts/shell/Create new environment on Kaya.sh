module avail
module load Anaconda3/2023.09
<<<<<<< HEAD
module load cuda/12.4
=======
>>>>>>> afa18c04fdf209d6b14f1c2be166149dd6dfc6a7
module list

conda init

# conda create --prefix /group/pmc021/amunif/conda_environments/epi-thesis
conda create -n epi-thesis python=3.11.8

# conda activate /group/pmc021/amunif/conda_environments/epi-thesis
conda activate epi-thesis

# Interactive session
sinfo
srun --time=00:30:00 --account=amunif --partition=gpu --gres=gpu:p100:1 --pty /bin/bash -l
salloc -w n003 -p gpu
nvidia-smi # check the GPU version

### Inside the GPU worker
module load Anaconda3/2023.09
conda create -n gpu_test python=3.11.5
conda activate gpu_test
pip install tensorflow[and-cuda]

### Remove Anaconda environment
conda env list
conda env remove --name myenv
conda env list