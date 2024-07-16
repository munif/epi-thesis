conda create -p /group/pmc021/amunif/env/pytorch python=3.11
chmod -R 775 /group/pmc021/amunif/env/pytorch/bin

# Logout, disconnect then login again
conda deactivate
module load Anaconda3/2024.06
module load cuda/11.8
conda activate /group/pmc021/amunif/env/pytorch/
which python
python -V

# Look for GPU support CUDA 12.4
conda search -c conda-forge pytorch=2.3.1

# conda install "conda-forge/linux-64::pytorch=2.3.1=cuda118_py311h0047a46_300"
# conda install "conda-forge/linux-64::pytorch=2.3.1=cuda120_py311hf6aebf0_300"

conda install "conda-forge/linux-64::pytorch 2.3.1 cuda118_py311h0047a46_300"
conda install "conda-forge/linux-64::pytorch=2.3.1=cuda120_py311hf6aebf0_300"


# https://stackoverflow.com/questions/74836151/nothing-provides-cuda-needed-by-tensorflow-2-10-0-cuda112py310he87a039-0
# CONDA_OVERRIDE_CUDA="11.8" conda install "conda-forge/linux-64::pytorch=2.3.1=cuda118_py311h0047a46_300"

conda install pandas polars-lts-cpu scikit-learn matplotlib numpy tensorboard tensorflow


# Remove the environment
conda deactivate
conda env remove -p /group/pmc021/amunif/env/pytorch


# Running Tensorboard
chmod 775 -Rf /group/pmc021/amunif/env/pytorch/
tensorboard --logdir=xxx