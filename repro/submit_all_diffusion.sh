#!/bin/bash
# Submit the diffusion-arm sweep: one job array (5 seeds) per system.
# Usage: ./submit_all_diffusion.sh [sysname ...]   (no args = all systems)
set -euo pipefail
ROOT=/home/x-cchen47/Analytical-Lyapunov-Function-Discovery
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_diffusion
cd "$ROOT"
WANT=("$@")
want() { [[ ${#WANT[@]} -eq 0 ]] && return 0; for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done; return 1; }

tail -n +2 repro/systems.tsv | while IFS=$'\t' read -r sys src bm dim wt; do
  [[ -z "${sys:-}" || "$sys" == \#* ]] && continue
  want "$sys" || continue
  [[ -d "$RUNS_BASE/$sys" ]] || { echo "MISSING run dir for $sys (run setup_runs_diffusion.sh)"; exit 1; }
  echo "[submit-diff] $sys benchmark=$bm walltime=$wt seeds=1..5"
  sbatch --job-name="lyad_$sys" --time="$wt" --array=0-4 \
         --export=ALL,SYS="$sys",BM="$bm" \
         --chdir="$RUNS_BASE/$sys" \
         "$ROOT/repro/run_one_diffusion.sbatch"
done
