#!/bin/bash
# Submit one SLURM job array (5 seeds) per system, with per-system walltime.
# Usage: ./submit_all.sh [sysname ...]   (no args = all systems)
set -euo pipefail
ROOT=/users/9/chen8596/LyaDis
cd "$ROOT"
WANT=("$@")
want() { [[ ${#WANT[@]} -eq 0 ]] && return 0; for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done; return 1; }

tail -n +2 repro/systems.tsv | while IFS=$'\t' read -r sys src bm dim wt; do
  [[ -z "${sys:-}" || "$sys" == \#* ]] && continue
  want "$sys" || continue
  [[ -d "repro/runs/$sys" ]] || { echo "MISSING run dir for $sys (run setup_runs.sh)"; exit 1; }
  echo "[submit] $sys benchmark=$bm walltime=$wt seeds=1..5"
  sbatch --job-name="lya_$sys" --time="$wt" --array=0-4 \
         --export=ALL,SYS="$sys",BM="$bm" \
         --chdir="$ROOT/repro/runs/$sys" \
         "$ROOT/repro/run_one.sbatch"
done
