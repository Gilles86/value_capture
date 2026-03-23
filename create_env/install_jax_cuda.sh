#!/bin/bash
#SBATCH --job-name=install_jax_cuda
#SBATCH --output=/home/gdehol/logs/install_jax_cuda_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Install JAX with CUDA 12 support into the value_capture conda environment.
# Must run on a GPU node so the CUDA libraries are visible during install.
#
# Usage:
#   sbatch create_env/install_jax_cuda.sh

. $HOME/init_conda.sh
module load gpu

echo "Installing jax[cuda12] into value_capture env..."
conda run -n value_capture pip install -U "jax[cuda12]"

echo ""
echo "Verifying JAX sees the GPU:"
conda run -n value_capture python -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
"
