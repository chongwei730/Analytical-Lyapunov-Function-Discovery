#!/bin/bash
#SBATCH --account=cis260360-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --output=%x_seed%a_%A.out
# Anvil (Purdue RCAC) port of run_one.sbatch.
# Submitted by submit_all_anvil.sh, which sets --job-name, --time, --array, and
# exports SYS (system run-dir name) and BM (benchmark entry).
# Array index = seed offset; seed = 1 + SLURM_ARRAY_TASK_ID  -> seeds 1..5.
# NOTE: no `set -u` — conda activate.d scripts reference unset vars.
set -eo pipefail
ROOT=/home/x-cchen47/Analytical-Lyapunov-Function-Discovery
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs
RUNDIR="$RUNS_BASE/$SYS"
SEED=$(( 1 + SLURM_ARRAY_TASK_ID ))

module load conda/2024.09 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alfd

cd "$RUNDIR"
export PYTHONPATH="$RUNDIR/libs/sd3/dso"
# Pin BLAS/numpy threads to 1: verification runs ~SLURM_CPUS_PER_TASK pool
# workers, so per-worker multithreading would re-oversubscribe the cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LYA_BENCHMARK="$BM"
export LYA_SEED_START="$SEED"
export LYA_SEED_RUNS=1
# full reproduction: do NOT set LYA_NSAMPLES (uses paper's 2,000,000)

echo "[job] sys=$SYS benchmark=$BM seed=$SEED host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
python -u Lyapunov_test_dso.py
echo "[job] done sys=$SYS seed=$SEED end=$(date -Is)"
