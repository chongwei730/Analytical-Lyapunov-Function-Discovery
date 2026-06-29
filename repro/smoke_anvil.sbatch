#!/bin/bash
#SBATCH --account=cis260360-gpu
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --job-name=lya_smoke
#SBATCH --output=%x_%j.out
# Smoke test: pendulum, 1 seed, reduced n_samples, on gpu-debug.
set -eo pipefail
ROOT=/home/x-cchen47/Analytical-Lyapunov-Function-Discovery
module load conda/2024.09 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alfd
cd "$ROOT"
export PYTHONPATH="$ROOT/libs/sd3/dso"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LYA_BENCHMARK=fn_d_all_y
export LYA_SEED_START=1
export LYA_SEED_RUNS=1
export LYA_NSAMPLES=20000   # tiny, just to confirm the pipeline runs end-to-end
echo "[smoke] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
python -c "import torch; print('[smoke] torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -u Lyapunov_test_dso.py
echo "[smoke] done $(date -Is)"
