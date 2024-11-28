conda create -p /group/pmc021/amunif/env/xgbenv python=3.11
chmod -R 775 /group/pmc021/amunif/env/xgbenv/
conda deactivate
conda activate /group/pmc021/amunif/env/xgbenv
conda install -c conda-forge pandas polars-lts-cpu scikit-learn matplotlib numpy xgboost jupyterlab ipykernel -y
conda install -c conda-forge nb_conda_kernels

conda remove sqlite libsqlite
conda update -all
conda install sqlite libsqlite