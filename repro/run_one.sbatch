#!/bin/bash
#SBATCH --account=jinzn
#SBATCH --partition=a100-4,a100-8,v100,msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --output=%x_seed%a_%A.out
# Submitted by submit_all.sh, which sets --job-name, --time, --array, and
# exports SYS (system run-dir name) and BM (benchmark entry).
# Array index = seed offset; seed = 1 + SLURM_ARRAY_TASK_ID  -> seeds 1..5.
# NOTE: no `set -u` — conda's activate.d scripts reference unset vars (e.g.
# MKL_INTERFACE_LAYER) and would abort under nounset.
set -eo pipefail
ROOT=/users/9/chen8596/LyaDis
RUNDIR="$ROOT/repro/runs/$SYS"
SEED=$(( 1 + SLURM_ARRAY_TASK_ID ))

source ~/miniconda3/etc/profile.d/conda.sh
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
