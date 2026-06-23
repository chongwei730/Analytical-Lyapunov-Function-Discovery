#!/bin/bash
#SBATCH --account=jinzn
#SBATCH --partition=a100-4,a100-8,v100,msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=00:30:00
#SBATCH --job-name=lya_vgpu
#SBATCH --output=%x_%j.out
# Quick GPU validation: exercises the cuda:1 PGD falsifier path on a 2-GPU node
# with reduced samples, before the full launch.
set -eo pipefail
ROOT=/users/9/chen8596/LyaDis
RUNDIR="$ROOT/repro/runs/pendulum"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alfd
cd "$RUNDIR"
export PYTHONPATH="$RUNDIR/libs/sd3/dso"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LYA_BENCHMARK=fn_d_all_y
export LYA_SEED_START=1
export LYA_SEED_RUNS=1
export LYA_NSAMPLES=3000
echo "[vgpu] host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true
python -u Lyapunov_test_dso.py
echo "[vgpu] done end=$(date -Is)"
