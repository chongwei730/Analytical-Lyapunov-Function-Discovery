#!/bin/bash
# Anvil port: submit one SLURM job array (5 seeds) per system.
# Usage: ./submit_all_anvil.sh [sysname ...]   (no args = all systems)
set -euo pipefail
ROOT=/home/x-cchen47/Analytical-Lyapunov-Function-Discovery
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs
cd "$ROOT"
WANT=("$@")
want() { [[ ${#WANT[@]} -eq 0 ]] && return 0; for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done; return 1; }

tail -n +2 repro/systems.tsv | while IFS=$'\t' read -r sys src bm dim wt; do
  [[ -z "${sys:-}" || "$sys" == \#* ]] && continue
  want "$sys" || continue
  [[ -d "$RUNS_BASE/$sys" ]] || { echo "MISSING run dir for $sys (run setup_runs_anvil.sh)"; exit 1; }
  echo "[submit] $sys benchmark=$bm walltime=$wt seeds=1..5"
  sbatch --job-name="lya_$sys" --time="$wt" --array=0-4 \
         --export=ALL,SYS="$sys",BM="$bm" \
         --chdir="$RUNS_BASE/$sys" \
         "$ROOT/repro/run_one_anvil.sbatch"
done
