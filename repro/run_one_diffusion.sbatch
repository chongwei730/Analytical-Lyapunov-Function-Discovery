#!/bin/bash
#SBATCH --account=cis260360-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --output=%x_seed%a_%A.out
# Diffusion-arm variant of run_one_anvil.sbatch. Submitted by submit_all_diffusion.sh
# which sets --job-name/--time/--array and exports SYS, BM. seed = 1 + array index.
set -eo pipefail
ROOT=/home/x-cchen47/Analytical-Lyapunov-Function-Discovery
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_diffusion
RUNDIR="$RUNS_BASE/$SYS"
SEED=$(( 1 + SLURM_ARRAY_TASK_ID ))

module load conda/2024.09 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alfd

cd "$RUNDIR"
export PYTHONPATH="$RUNDIR/libs/sd3/dso"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LYA_BENCHMARK="$BM"
export LYA_SEED_START="$SEED"
export LYA_SEED_RUNS=1
export LYA_CONTROLLER=DiffusionController
export LYA_LAMBDA="${LYA_LAMBDA:-0.5}"
# full reproduction: do NOT set LYA_NSAMPLES (uses paper's 2,000,000)

echo "[diff] sys=$SYS benchmark=$BM seed=$SEED lambda=${LYA_LAMBDA:-0.5} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
python -u Lyapunov_test_dso.py
echo "[diff] done sys=$SYS seed=$SEED end=$(date -Is)"
