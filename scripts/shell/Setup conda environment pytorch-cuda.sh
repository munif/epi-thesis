which python

python -V

module load Anaconda3/2024.06

module load cuda/11.8

CONDA_OVERRIDE_CUDA="11.8" conda install "conda-forge/linux-64::pytorch=2.3.1=cuda118_py311h0047a46_300"