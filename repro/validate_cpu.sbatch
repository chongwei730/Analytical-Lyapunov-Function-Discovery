#!/bin/bash
#SBATCH --account=jinzn
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --job-name=lya_validate
#SBATCH --output=%x_%j.out
# Fast end-to-end pipeline check on CPU (short queue), reduced n_samples.
# Confirms conda env + dso shadowing + config + task + training loop all run
# to completion inside the SLURM environment before the big GPU launch.
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
export LYA_NSAMPLES=3000   # tiny: ~6 batches, just to reach a TEST RESULT
echo "[validate] host=$(hostname) start=$(date -Is)"
python -u Lyapunov_test_dso.py
echo "[validate] done end=$(date -Is)"
